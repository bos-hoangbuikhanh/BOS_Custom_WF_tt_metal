# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import List

from loguru import logger

import ttnn

from .Config import CORE_SET_14, CORE_SET_16, LAYER_CONFIG
from .Conv import convNxN
from .Linear import ResnetLinear


def conv1x1(
    x,
    device,
    input_height,
    input_width,
    in_channels,
    out_channels,
    batch_size,
    weight_tensor,
    bias_tensor,
    conv_config,
    compute_config,
    dtype,
    return_output_dim,
    return_weights_and_bias=True,
    kernel_size=(1, 1),
    stride=(1, 1),
    padding=(0, 0),
):
    """1x1 convolution"""
    return convNxN(
        x,
        device,
        input_height,
        input_width,
        in_channels,
        out_channels,
        batch_size,
        weight_tensor,
        bias_tensor,
        conv_config,
        compute_config,
        dtype,
        return_output_dim,
        return_weights_and_bias,
        kernel_size,
        stride,
        padding,
    )


def conv3x3(
    x,
    device,
    input_height,
    input_width,
    in_channels,
    out_channels,
    batch_size,
    weight_tensor,
    bias_tensor,
    conv_config,
    compute_config,
    dtype,
    return_output_dim,
    return_weights_and_bias=True,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding=(1, 1),
):
    """3x3 convolution"""
    return convNxN(
        x,
        device,
        input_height,
        input_width,
        in_channels,
        out_channels,
        batch_size,
        weight_tensor,
        bias_tensor,
        conv_config,
        compute_config,
        dtype,
        return_output_dim,
        return_weights_and_bias,
        kernel_size,
        stride,
        padding,
    )


ops_parallel_config = {"layer1_module1_input": None}


class Bottleneck:
    expansion: int = 4

    def __init__(self, parameters, downsample, stride, model_config) -> None:
        # init is just to pre-process pytorch weights and bias tensors
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        assert self.conv1_weight_tensor.shape[2] == 1

        self.conv2_weight_tensor = parameters.conv2.weight
        self.conv2_bias_tensor = parameters.conv2.bias
        self.conv2_input_channels = self.conv2_weight_tensor.shape[1]
        self.conv2_output_channels = self.conv2_weight_tensor.shape[0]
        self.conv2_stride = 2 if downsample else 1
        assert self.conv2_weight_tensor.shape[2] == 3

        self.conv3_weight_tensor = parameters.conv3.weight
        self.conv3_bias_tensor = parameters.conv3.bias
        self.conv3_input_channels = self.conv3_weight_tensor.shape[1]
        self.conv3_output_channels = self.conv3_weight_tensor.shape[0]
        assert self.conv3_weight_tensor.shape[2] == 1

        self.downsample = downsample
        self.stride = stride
        if downsample:
            self.ds_conv_weight_tensor = parameters.downsample.weight
            self.ds_conv_bias_tensor = parameters.downsample.bias
            self.ds_conv_input_channels = self.ds_conv_weight_tensor.shape[1]
            self.ds_conv_output_channels = self.ds_conv_weight_tensor.shape[0]
            assert self.ds_conv_weight_tensor.shape[2] == 1
        self.model_config = model_config
        return

    def __call__(self, x, device, batch_size, input_height, input_width, confs, layer_module=None):
        (
            reshard_if_not_optimal,
            height_sharding,
            packer_l1_acc,
            enable_act_double_buffer,
            force_split_reader,
            ops_parallel_config,
        ) = confs
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=self.model_config["MATH_FIDELITY"], packer_l1_acc=packer_l1_acc
        )

        logger.debug(
            f"==== Running {batch_size}, {input_height}, {input_width}, {self.conv1_input_channels}, {self.conv1_output_channels}"
        )
        module_input_height, module_input_width = input_height, input_width
        ds_input_height, ds_input_width = input_height, input_width

        # conv1 is 1x1 conv
        logger.debug(f"Running conv1")
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            if height_sharding
            else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            reshard_if_not_optimal=reshard_if_not_optimal,
        )
        out, [input_height, input_width], [self.conv1_weight_tensor, self.conv1_bias_tensor] = conv1x1(
            x,
            device,
            input_height,
            input_width,
            self.conv1_input_channels,
            self.conv1_output_channels,
            batch_size,
            self.conv1_weight_tensor,
            self.conv1_bias_tensor,
            conv_config,
            self.compute_config,
            self.model_config["ACTIVATIONS_DTYPE"],
            True,
            True,
        )

        run_downsample_before_conv2 = False
        ds_out = None

        if run_downsample_before_conv2:
            if layer_module and layer_module == "layer4_module1":
                if ops_parallel_config and "layer4_module1_downsample" in ops_parallel_config:
                    x = ttnn.to_memory_config(x, ops_parallel_config["layer4_module1_downsample"])
            ds_out = self.run_downsample_if_req(
                x,
                device,
                batch_size,
                ds_input_height,
                ds_input_width,
                reshard_if_not_optimal,
                height_sharding,
                packer_l1_acc,
                False,
                force_split_reader,
            )
            if layer_module and layer_module == "layer4_module1":
                if ops_parallel_config and "layer4_module1_downsample" not in ops_parallel_config:
                    x_memory_config = ttnn.get_memory_config(ds_out)
                    ops_parallel_config["layer4_module1_downsample"] = ttnn.create_sharded_memory_config_(
                        ttnn.Shape([batch_size, ds_input_height, ds_input_width, self.conv1_input_channels]),
                        x_memory_config.shard_spec.grid,
                        x_memory_config.memory_layout,
                        x_memory_config.shard_spec.orientation,
                        tile_layout=True,
                    )

        if layer_module and layer_module == "layer4_module1":
            if ops_parallel_config and "layer4_module1_input" in ops_parallel_config:
                out = ttnn.to_memory_config(out, ops_parallel_config["layer4_module1_input"])

        # conv2 is 3x3 conv
        logger.debug(f"Running conv2")
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            reallocate_halo_output=False,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            if height_sharding
            else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            reshard_if_not_optimal=reshard_if_not_optimal,
            enable_act_double_buffer=enable_act_double_buffer,
            enable_weights_double_buffer=True,
            force_split_reader=force_split_reader,
        )
        out, [input_height, input_width], [self.conv2_weight_tensor, self.conv2_bias_tensor] = conv3x3(
            out,
            device,
            input_height,
            input_width,
            self.conv2_input_channels,
            self.conv2_output_channels,
            batch_size,
            self.conv2_weight_tensor,
            self.conv2_bias_tensor,
            conv_config,
            self.compute_config,
            self.model_config["ACTIVATIONS_DTYPE"],
            True,
            True,
            stride=(self.stride, self.stride),
        )

        if layer_module and layer_module == "layer4_module1":
            if ops_parallel_config and "layer4_module1_input" not in ops_parallel_config:
                x_memory_config = ttnn.get_memory_config(out)
                sharded_config = ttnn.create_sharded_memory_config_(
                    ttnn.Shape([batch_size, module_input_height, module_input_width, self.conv2_input_channels]),
                    x_memory_config.shard_spec.grid,
                    x_memory_config.memory_layout,
                    x_memory_config.shard_spec.orientation,
                    tile_layout=True,
                )
                ops_parallel_config["layer4_module1_input"] = sharded_config

        # conv3 is 1x1 conv
        logger.debug(f"Running conv3")
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            if height_sharding
            else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            reshard_if_not_optimal=reshard_if_not_optimal,
        )
        out, [self.conv3_weight_tensor, self.conv3_bias_tensor] = conv1x1(
            out,
            device,
            input_height,
            input_width,
            self.conv3_input_channels,
            self.conv3_output_channels,
            batch_size,
            self.conv3_weight_tensor,
            self.conv3_bias_tensor,
            conv_config,
            self.compute_config,
            self.model_config["ACTIVATIONS_DTYPE"],
            False,
            True,
        )

        if not run_downsample_before_conv2:
            ds_out = self.run_downsample_if_req(
                x,
                device,
                batch_size,
                ds_input_height,
                ds_input_width,
                reshard_if_not_optimal,
                height_sharding,
                packer_l1_acc,
                enable_act_double_buffer,
                force_split_reader,
            )

        assert ds_out is not None, "ds_out is None"
        if (batch_size in [1, 2] and "layer2" in layer_module) or (batch_size == 1 and "layer4" in layer_module):
            ds_out = ttnn.to_memory_config(ds_out, out.memory_config())
        if ttnn.get_memory_config(out) != ttnn.get_memory_config(ds_out):
            out = ttnn.to_memory_config(out, ds_out.memory_config())
        assert ttnn.get_memory_config(out) == ttnn.get_memory_config(
            ds_out
        ), f"{ttnn.get_memory_config(out)} != {ttnn.get_memory_config(ds_out)}"

        out = ttnn.add_(out, ds_out, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)])
        ttnn.deallocate(ds_out)

        return out, input_height, input_width

    def run_downsample_if_req(
        self,
        x,
        device,
        batch_size,
        input_height,
        input_width,
        reshard_if_not_optimal=False,
        height_sharding=None,
        packer_l1_accum_enabled=True,
        enable_act_double_buffer=False,
        force_split_reader=False,
    ):
        if self.downsample:
            logger.debug(f"Running downsample")
            conv_config = ttnn.Conv2dConfig(
                weights_dtype=self.model_config["WEIGHTS_DTYPE"],
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if height_sharding
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                deallocate_activation=True,
                reallocate_halo_output=True,
                reshard_if_not_optimal=reshard_if_not_optimal,
                enable_act_double_buffer=enable_act_double_buffer
                if height_sharding
                else True
                if input_width < 56
                else False,
                enable_weights_double_buffer=True if input_width < 56 else False,
                force_split_reader=force_split_reader,
            )
            ds_out, [self.ds_conv_weight_tensor, self.ds_conv_bias_tensor] = convNxN(
                x,
                device,
                input_height,
                input_width,
                self.ds_conv_input_channels,
                self.ds_conv_output_channels,
                batch_size,
                self.ds_conv_weight_tensor,
                self.ds_conv_bias_tensor,
                conv_config,
                ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=self.model_config["MATH_FIDELITY"],
                    packer_l1_acc=packer_l1_accum_enabled,
                ),
                self.model_config["ACTIVATIONS_DTYPE"],
                False,
                True,
                (1, 1),
                (self.stride, self.stride),
                (0, 0),
            )
            ttnn.deallocate(x)
            ds_out = ttnn.reallocate(ds_out)
        else:
            ds_out = x
        return ds_out


class resnet50:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        model_config,
        input_shape,
        kernel_size,
        stride,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
    ) -> None:
        super().__init__()
        self.num_classes = 1000
        self.device = device
        self.conv_input_face_shape_hw = [224, 224]
        self.batch_size = batch_size
        self.model_config = model_config
        self.inplanes = 64
        self.final_output_mem_config = final_output_mem_config

        # Define Model's Layer
        layers = [3, 4, 6, 3]
        self.conv1_weight_tensor = parameters.conv1.weight
        self.conv1_bias_tensor = parameters.conv1.bias
        assert layers == [3, 4, 6, 3]
        assert self.conv1_weight_tensor.shape[2] == 4

        self.conv1_input_channels = self.conv1_weight_tensor.shape[1]
        self.conv1_output_channels = self.conv1_weight_tensor.shape[0]
        self.conv1_kernel_size = (4, 4)
        self.conv1_stride = (1, 1)
        self.conv1_padding = (0, 0)
        self.conv1_input_height = 115
        self.conv1_input_width = 115
        self.conv1_output_height = (
            (self.conv1_input_height - self.conv1_kernel_size[0] + 2 * self.conv1_padding[0]) // self.conv1_stride[0]
        ) + 1
        self.conv1_output_width = (
            (self.conv1_input_width - self.conv1_kernel_size[1] + 2 * self.conv1_padding[1]) // self.conv1_stride[1]
        ) + 1
        self.conv1_config = ttnn.Conv2dConfig(
            weights_dtype=self.model_config["WEIGHTS_DTYPE"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=dealloc_input,
            act_block_h_override=32,
            enable_act_double_buffer=True,
            force_split_reader=True,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=False,
        )
        self.conv1_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=self.model_config["MATH_FIDELITY"],
            packer_l1_acc=True,
        )
        self.layer_component = {
            "layer1": {
                "module": self._make_layer(
                    parameters=parameters.layer1, planes=64, blocks=layers[0], stride=1, model_config=model_config
                )
            },
            "layer2": {
                "module": self._make_layer(
                    parameters=parameters.layer2, planes=128, blocks=layers[1], stride=2, model_config=model_config
                )
            },
            "layer3": {
                "module": self._make_layer(
                    parameters=parameters.layer3, planes=256, blocks=layers[2], stride=2, model_config=model_config
                )
            },
            "layer4": {
                "module": self._make_layer(
                    parameters=parameters.layer4, planes=512, blocks=layers[3], stride=2, model_config=model_config
                )
            },
        }
        self.avgpool = ttnn.global_avg_pool2d
        self.fc_weight_tensor = parameters.fc.weight
        self.fc_bias_tensor = parameters.fc.bias
        self.fc = ResnetLinear(
            512 * Bottleneck.expansion,
            1024,
            ttnn.to_device(self.fc_weight_tensor, device),
            ttnn.to_device(self.fc_bias_tensor, device),
            ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            model_config,
            self.device,
            batch_size,
            ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
        )

    def __call__(self, input_tensor, device, ops_parallel_config) -> ttnn.Tensor:
        self.is_first_run = False
        if not ops_parallel_config:
            self.is_first_run = True
            logger.debug(f"==== First run")
        else:
            logger.debug(f"==== Optimized run")

        # run fold
        logger.debug(f"==== fold on host")
        fold_output_tensor = input_tensor
        n, c, h, w = fold_output_tensor.shape
        fold_output_tensor = ttnn.reshape(fold_output_tensor, (1, 1, n * c * h, w))

        # first conv (ReLU is fused with conv1)
        logger.debug(f"==== first conv")
        x, [x_height, x_width], [self.conv1_weight_tensor, self.conv1_bias_tensor] = convNxN(
            fold_output_tensor,
            device,
            self.conv1_input_height,
            self.conv1_input_width,
            self.conv1_input_channels,
            self.conv1_output_channels,
            self.batch_size,
            self.conv1_weight_tensor,
            self.conv1_bias_tensor,
            self.conv1_config,
            self.conv1_compute_config,
            self.model_config["ACTIVATIONS_DTYPE"],
            True,
            True,
            self.conv1_kernel_size,
            self.conv1_stride,
            self.conv1_padding,
        )
        ttnn.deallocate(fold_output_tensor)

        # MaxPool
        logger.debug(f"==== maxpool")
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=x_height,
            input_w=x_width,
            channels=self.conv1_output_channels,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
        )

        x_height, x_width = 56, 56
        x = ttnn.reshape(x, (1, 1, x_height * x_width * self.batch_size, 64))
        x = ttnn.to_memory_config(
            x,
            ttnn.create_sharded_memory_config_(
                ttnn.Shape([self.batch_size * x_height * x_width, 64]),
                CORE_SET_14,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
            ),
        )
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=self.model_config["ACTIVATIONS_DTYPE"])

        # layer1
        x, x_height, x_width = self._forward_layer(x, "layer1", self.batch_size, x_height, x_width, device)

        # layer2
        x, x_height, x_width = self._forward_layer(x, "layer2", self.batch_size, x_height, x_width, device)

        # layer3
        x, x_height, x_width = self._forward_layer(x, "layer3", self.batch_size, x_height, x_width, device)

        # layer4
        x, x_height, x_width = self._forward_layer(x, "layer4", self.batch_size, x_height, x_width, device)

        grid_size = (4, 4)
        shard_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1))}
        )
        shard_shape = [x.volume() // x.padded_shape[-1], x.padded_shape[-1] // (grid_size[0] * grid_size[1])]
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        width_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
        )
        x = ttnn.to_memory_config(x, width_sharded_mem_config)

        unpadded_shape = x.shape
        output_tensor_end_shape = [
            unpadded_shape[0] - 1,
            unpadded_shape[1] - 1,
            unpadded_shape[2] - 1,
            unpadded_shape[3] - 1,
        ]
        x = ttnn.untilize_with_unpadding(
            x, output_tensor_end=output_tensor_end_shape, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        )
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, (self.batch_size, x.shape[1], x.shape[2] // self.batch_size, x.shape[3]))
        x = ttnn.to_memory_config(
            x,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_grid,
                    [x.volume() // x.shape[-1], x.shape[-1] // shard_grid.num_cores()],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )

        unpadded_shape = x.padded_shape
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            math.ceil(unpadded_shape[2] / 32) * 32,
            math.ceil(unpadded_shape[3] / 32) * 32,
        ]
        x = ttnn.tilize_with_val_padding(
            x,
            padded_shape,
            0.0,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug(f"==== avgpool")
        x = self.avgpool(x, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

        output_tensor_end_shape = [x.padded_shape[0] - 1, x.padded_shape[1] - 1, 1 - 1, x.padded_shape[3] - 1]
        x = ttnn.untilize_with_unpadding(
            x, output_tensor_end=output_tensor_end_shape, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        )
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, (1, x.padded_shape[1], self.batch_size * x.padded_shape[2], x.padded_shape[3]))
        x = ttnn.to_memory_config(
            x,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_grid,
                    [x.volume() // x.shape[-1], x.shape[-1] // shard_grid.num_cores()],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )

        unpadded_shape = x.padded_shape
        padded_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            math.ceil(unpadded_shape[2] / 32) * 32,
            math.ceil(unpadded_shape[3] / 32) * 32,
        ]
        x = ttnn.tilize_with_val_padding(
            x,
            padded_shape,
            0.0,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=self.model_config["ACTIVATIONS_DTYPE"],
        )

        logger.debug(f"==== fc")
        x = self.fc(x)

        desired_shape = list(x.shape)
        desired_shape[-1] = self.num_classes
        output_tensor_end_shape = [
            desired_shape[0] - 1,
            desired_shape[1] - 1,
            desired_shape[2] - 1,
            desired_shape[3] - 1,
        ]
        x = ttnn.untilize_with_unpadding(
            x, output_tensor_end=output_tensor_end_shape, memory_config=self.final_output_mem_config
        )
        x = ttnn.reshape(x, (self.batch_size, x.shape[1], x.shape[2] // self.batch_size, x.shape[3]))

        return x

    def _make_layer(self, parameters, planes: int, blocks: int, stride: int, model_config=None) -> List[Bottleneck]:
        layers = []
        layers.append(
            Bottleneck(
                parameters=parameters[0],
                downsample=stride != 1 or self.inplanes != planes * Bottleneck.expansion,
                stride=stride,
                model_config=model_config,
            )
        )
        self.inplanes = planes * Bottleneck.expansion
        for block_num in range(1, blocks):
            layers.append(
                Bottleneck(parameters=parameters[block_num], downsample=False, stride=1, model_config=model_config)
            )
        return layers

    def _forward_layer(self, input_tensor, layer_name, batch_size, x_height, x_width, device) -> ttnn.Tensor:
        layer_module1_input_shape = ttnn.Shape(input_tensor.padded_shape)
        if layer_name in ["layer1", "layer2", "layer3"]:
            x = input_tensor
        elif layer_name == "layer4":
            x = ttnn.to_memory_config(
                input_tensor,
                ttnn.create_sharded_memory_config_(
                    layer_module1_input_shape,
                    CORE_SET_16,
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    tile_layout=True,
                ),
            )
        for i in range(len(LAYER_CONFIG[layer_name])):
            logger.debug(f"==== Running {layer_name} module{i+1}")
            x, x_height, x_width = self.layer_component[layer_name]["module"][i](
                x,
                device,
                batch_size,
                x_height,
                x_width,
                LAYER_CONFIG[layer_name][f"module{i+1}"],
                layer_module=f"{layer_name}_module{i+1}",
            )
            if i == 0 and self.is_first_run and not (layer_name == "layer4"):
                x_memory_config = ttnn.get_memory_config(x)
                ops_parallel_config["layer1_module1_input"] = ttnn.create_sharded_memory_config_(
                    layer_module1_input_shape,
                    x_memory_config.shard_spec.grid,
                    x_memory_config.memory_layout,
                    x_memory_config.shard_spec.orientation,
                    tile_layout=True,
                )
        return x, x_height, x_width

    def __del__(self):
        # Nothing to do
        pass


class ModelRunnerWrapper:
    def __init__(
        self,
        device,
        batch_size,
        test_infra,
        use_trace=False,
        use_2cq=False,
        run_warmup=True,
    ):
        from models.bos_model.resnet50.tests.resnet50_test_infra import create_test_infra

        self.device = device
        self.batch_size = batch_size
        self.test_infra = create_test_infra(
            self.device,
            self.batch_size,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
            True,
            True,
            ttnn.L1_MEMORY_CONFIG,
            None,
        )
        self.input_mem_config = None
        self.use_trace = use_trace
        self.use_2cq = use_2cq
        if self.use_2cq:
            self.op_event = None
            self.write_event = None
            self.tt_image_res = None
            self.input_tensor = None
            self.sharded_mem_config_DRAM = None
        run_warmup and self._warmup()

    def __call__(self, pixel_values):
        tt_inputs_host = self._prepare_inputs(pixel_values)
        self._inference(tt_inputs_host)
        tt_out = ttnn.reshape(self.tt_output_res, (self.batch_size, 1, 1000))
        return tt_out

    def _inference(self, tt_inputs_host):
        if self.use_2cq:
            ttnn.wait_for_event(1, self.op_event)
            ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
            self.write_event = ttnn.record_event(self.device, 1)
            ttnn.wait_for_event(0, self.write_event)
            if self.use_trace:
                self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
                self.op_event = ttnn.record_event(self.device, 0)
                ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
            else:
                self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
                self.op_event = ttnn.record_event(self.device, 0)
                self.tt_output_res = self.test_infra.run()
        else:
            if self.use_trace:
                ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res)
                ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
            else:
                self.test_infra.input_tensor = tt_inputs_host.to(self.device, self.input_mem_config)
                self.tt_output_res = self.test_infra.run()

    def _prepare_inputs(self, pixel_values):
        if self.use_2cq:
            (
                tt_inputs_host,
                self.sharded_mem_config_DRAM,
                self.input_mem_config,
            ) = self.test_infra.setup_dram_sharded_input(self.device, pixel_values)
        else:
            tt_inputs_host, self.input_mem_config = self.test_infra.setup_l1_sharded_input(self.device, pixel_values)
        return tt_inputs_host

    def _transfer_and_reshard_2cq(self, tt_inputs_host):
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)

    def _setup_input_tensor(self, tt_inputs_host):
        if self.use_2cq:
            self._transfer_and_reshard_2cq(tt_inputs_host)
        else:
            self.test_infra.input_tensor = tt_inputs_host.to(self.device, self.input_mem_config)

    def _create_dummy_input(self):
        tt_inputs_host = self._prepare_inputs(
            ttnn.to_torch(
                ttnn.zeros(
                    [self.batch_size, 3, 224, 224],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                )
            )
        )
        return tt_inputs_host

    def _warmup(self):
        tt_inputs_host = self._create_dummy_input()

        # First run configures convs JIT
        if self.use_2cq:
            self.tt_image_res = tt_inputs_host.to(self.device, self.sharded_mem_config_DRAM)
            self.op_event = ttnn.record_event(self.device, 0)
            self._transfer_and_reshard_2cq(tt_inputs_host)
            if self.use_trace:
                self.spec = self.test_infra.input_tensor.spec
            self.op_event = ttnn.record_event(self.device, 0)
        else:
            self.test_infra.input_tensor = tt_inputs_host.to(self.device, self.input_mem_config)
        _ = ttnn.from_device(self.test_infra.run(), blocking=True)
        self.test_infra.output_tensor.deallocate(force=True)

        # Optimized run
        self._setup_input_tensor(tt_inputs_host)
        self.test_infra.output_tensor.deallocate(force=True)
        _ = ttnn.from_device(self.test_infra.run(), blocking=True)

    def trace_capture(self):
        tt_inputs_host = self._create_dummy_input()
        self._setup_input_tensor(tt_inputs_host)
        self.test_infra.output_tensor.deallocate(force=True)
        trace_input_addr = self.test_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.tt_output_res = self.test_infra.run()
        if self.use_2cq:
            self.input_tensor = ttnn.allocate_tensor_on_device(self.spec, self.device)
            assert trace_input_addr == self.input_tensor.buffer_address()
        else:
            spec = self.test_infra.input_tensor.spec
            self.tt_image_res = ttnn.allocate_tensor_on_device(spec, self.device)
            assert trace_input_addr == self.tt_image_res.buffer_address()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)


class BenchmarkModelRunnerWrapper(ModelRunnerWrapper):
    def __init__(self, device, batch_size, test_infra, use_trace, use_2cq):
        super().__init__(device, batch_size, test_infra, use_trace, use_2cq, False)
        self.num_measurement_iterations = 1
        self._initialize_profiler()
        self._warmup()

    def __call__(self, pixel_values):
        self.profiler.start("inference_batch")
        tt_inputs_host = self._prepare_inputs(pixel_values)
        for _ in range(self.num_measurement_iterations):
            self._inference(tt_inputs_host)
            tt_out = ttnn.reshape(self.tt_output_res, (self.batch_size, 1, 1000))
        tt_out = ttnn.from_device(tt_out, blocking=True)
        self.profiler.end("inference_batch")
        return tt_out

    def _warmup(self):
        self.profiler.start("warmup")
        super()._warmup()
        self.profiler.end("warmup")

    def _initialize_profiler(self):
        from models.common.utility_functions import profiler

        self.profiler = profiler

    def set_num_measurement_iterations(self, val):
        self.num_measurement_iterations = val
