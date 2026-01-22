import collections.abc

import torch
import torch.nn as nn

import ttnn
from models.bos_model.yolov8s.utilities.utility_functions import weights_dict  # remove if not checking Golden
from models.bos_model.yolov8s.utilities.utility_functions import (
    _nearest_32,
    comp_pcc,
    prepare_conv_weights,
    tt_to_torch_tensor,
)


class GoldenConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=True, use_batchnorm=True):
        super(GoldenConv2D, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
            self.bn.eval()

    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x


class BOS_TTNN_Conv(nn.Module):
    def __init__(
        self,
        image_shape,
        in_channels,
        out_channels,
        base_address,
        device,
        kernel=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        batchnorm=True,
        activation=True,
        return_input=False,
        return_weight=False,
        layer_configs={},
        pcc_check=None,
    ):
        super().__init__()

        self.device = device
        self.image_shape = (
            image_shape if isinstance(image_shape, collections.abc.Iterable) else [image_shape, image_shape]
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel if isinstance(kernel, collections.abc.Iterable) else (kernel, kernel)
        self.stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        self.padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, collections.abc.Iterable) else (dilation, dilation)
        self.groups = groups
        self.use_bias = bias
        self.use_batchnorm = batchnorm
        self.base_address = base_address
        self.pcc_check = pcc_check

        self.act = activation
        self.return_weight = return_weight
        self.return_input = return_input

        self.conv_weights_address = self.base_address + "conv"
        # if self.use_batchnorm:
        #     self.weight, self.bias = prepare_conv_weights(self.base_address)
        # else:
        #     self.weight, self.bias = weights_dict[self.base_address + "conv.weight"], torch.zeros(
        #         self.out_channels, dtype=torch.bfloat16
        #     )
        self.weight, self.bias = prepare_conv_weights(self.base_address, not self.use_batchnorm)
        self.ttnn_weight = ttnn.from_torch(self.weight, ttnn.bfloat16)
        self.ttnn_bias = ttnn.from_torch(self.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), ttnn.bfloat16)
        self.activation = ttnn.silu
        self.conv_config = layer_configs[self.conv_weights_address]

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        if self.pcc_check is not None:
            self.golden_conv = GoldenConv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel,
                stride=self.stride,
                padding=self.padding,
                use_bias=not self.use_batchnorm,
                use_batchnorm=self.use_batchnorm,
            )
            if self.use_batchnorm:
                self.golden_conv.conv.weight = nn.Parameter(weights_dict[self.base_address + "conv.weight"])
            else:
                self.golden_conv.conv.weight, self.golden_conv.conv.bias = nn.Parameter(
                    weights_dict[self.base_address + "weight"]
                ), nn.Parameter(weights_dict[self.base_address + "bias"])
            if self.base_address == "model.0.":
                self.golden_conv.conv.weight = nn.Parameter(
                    nn.functional.pad(weights_dict[self.base_address + "conv.weight"], pad=(0, 0, 0, 0, 0, 29))
                )
            if self.use_batchnorm:
                self.golden_conv.bn.weight = nn.Parameter(weights_dict[self.base_address + "bn.weight"])
                self.golden_conv.bn.bias = nn.Parameter(weights_dict[self.base_address + "bn.bias"])
                self.golden_conv.bn.running_mean = weights_dict[self.base_address + "bn.running_mean"]
                self.golden_conv.bn.running_var = weights_dict[self.base_address + "bn.running_var"]
                self.golden_conv.bn.eps = 0.001
                self.golden_conv.bn.eval()

    def forward(self, x, interleave_if_optimal=False):
        if self.pcc_check is not None:
            torch_input = tt_to_torch_tensor(x)[:, :, : self.image_shape[0] * self.image_shape[1], :]
            torch_input = torch_input.permute(0, 3, 1, 2)
            torch_input = torch_input.reshape(1, self.in_channels, self.image_shape[0], self.image_shape[1])
            torch_input = torch_input.to(torch.float32)

        if interleave_if_optimal:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        # print(f"Base Address: {self.base_address}")
        # print(f"Input Shape: {x.shape}, Weight shape: {self.ttnn_weight.shape}")
        # print(f"Kernel: {self.kernel}, Stride: {self.stride}, Padding: {self.padding}, Dilation: {self.dilation}")
        # print(f"Input Sizes: {self.image_shape[0]}x{self.image_shape[1]}")
        # print(f"Input Memory Config: {x.memory_config()}")
        # print(f"Conv Config: {self.conv_config}")
        [output, [feature_height, feature_width], [self.ttnn_weight, self.ttnn_bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.ttnn_weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.ttnn_bias,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=1,
            input_height=self.image_shape[0],
            input_width=self.image_shape[1],
            conv_config=self.conv_config,
            # conv_op_cache={},
            compute_config=self.compute_config,
            # debug=False,
            # groups=self.groups,
            # memory_config=None,
            # slice_config=None,
            return_weights_and_bias=True,
            return_output_dim=True,
        )
        # print()

        if self.pcc_check is not None:
            torch_output = tt_to_torch_tensor(output)[:, :, : feature_height * feature_width, :]
            golden_output = self.golden_conv(torch_input).permute(0, 2, 3, 1)
            golden_output = golden_output.reshape(1, 1, feature_height * feature_width, self.out_channels)
            # feature_size = (32 - (feature_height * feature_width) % 32) % 32
            # golden_output = nn.functional.pad(golden_output, (0, 0, 0, feature_size))
            if golden_output.shape[-1] != torch_output.shape[-1]:
                golden_output = nn.functional.pad(golden_output, (0, torch_output.shape[-1] - golden_output.shape[-1]))
            _, pcc = comp_pcc(golden_output, torch_output)
            if self.return_weight:
                torch.save(torch_input, "x1.pt")
                torch.save(torch_output, "y1.pt")
            print(
                ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",
                self.base_address,
                "Convolution PCC =",
                pcc,
            )
            assert pcc >= self.pcc_check

            torch_input = torch_output.to(torch.float32)

        if self.act:
            output = self.activation(output)

            if self.pcc_check is not None:
                torch_output = tt_to_torch_tensor(output)[:, :, : feature_height * feature_width, :]
                golden_output = torch.nn.functional.silu(torch_input)
                _, pcc = comp_pcc(golden_output, torch_output)
                print(
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",
                    self.base_address,
                    "Silu PCC =",
                    pcc,
                )
                # assert(pcc >= self.pcc_check)
                assert pcc >= 0.97

        if self.return_weight:
            print("Conv_output: ", output)
            return x, output, self.ttnn_weight

        if not self.return_input:
            return output
        else:
            return x, output


class BOS_TTNN_MaxPool(nn.Module):
    def __init__(
        self,
        device,
        base_address,
        kernel=5,
        stride=1,
        padding=2,
        dilation=1,
        bias=False,
        layer_configs={},
        pcc_check=None,
    ):
        super().__init__()

        self.device = device
        self.kernel = kernel if isinstance(kernel, collections.abc.Iterable) else (kernel, kernel)
        self.stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        self.padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, collections.abc.Iterable) else (dilation, dilation)
        self.base_address = base_address
        self.pcc_check = pcc_check

        self.image_shape = None
        self.channels = None

        assert (not bias, "Bias for this Maxpool not implemented")

    def Golden_MaxPool2d(self, input_tensor, kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False):
        output = nn.functional.max_pool2d(
            input_tensor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

        return output

    def forward(self, x, image_shape):
        self.image_shape = image_shape
        self.channels = x.shape[-1]

        if self.pcc_check is not None:
            torch_input = tt_to_torch_tensor(x)[:, :, : self.image_shape[0] * self.image_shape[1], :]
            torch_input = (
                torch_input.permute(0, 3, 1, 2)
                .reshape(1, self.channels, self.image_shape[0], self.image_shape[1])
                .to(torch.float32)
            )

        output = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=1,
            input_h=image_shape[0],
            input_w=image_shape[1],
            channels=x.shape[-1],
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            # device=self.device, #NOTE: v0.53.0 no need passing device
        )

        if self.pcc_check is not None:
            torch_output = tt_to_torch_tensor(output)[:, :, : self.image_shape[0] * self.image_shape[1], :]
            golden_output = self.Golden_MaxPool2d(torch_input).permute(0, 2, 3, 1)
            golden_output = golden_output.reshape(1, 1, self.image_shape[0] * self.image_shape[1], self.channels)
            # feature_size = (32 - (self.image_shape[0] * self.image_shape[1]) % 32) % 32
            # golden_output = nn.functional.pad(golden_output, (0, 0, 0, feature_size))
            _, pcc = comp_pcc(golden_output, torch_output)
            print(
                ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",
                self.base_address,
                "MaxPool PCC =",
                pcc,
            )
            assert pcc >= self.pcc_check

        return output


class BOS_TTNN_Bottleneck(nn.Module):
    def __init__(
        self,
        image_shape,
        in_channels,
        out_channels,
        base_address,
        device,
        use_shortcut=True,
        groups=1,
        kernel=(3, 3),
        stride=1,
        padding=1,
        expansion=0.5,
        layer_configs={},
        pcc_check=None,
    ):
        super().__init__()

        self.device = device
        self.image_shape = image_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = int(out_channels * expansion)
        self.cv1_kernel = (kernel[0], kernel[0])
        self.cv2_kernel = (kernel[1], kernel[1])
        self.stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        self.padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)
        self.base_address = base_address
        self.pcc_check = pcc_check

        self.conv1 = BOS_TTNN_Conv(
            self.image_shape,
            self.in_channels,
            self.hidden_channels,
            self.base_address + "cv1.",
            self.device,
            self.cv1_kernel,
            self.stride,
            self.padding,
            layer_configs=layer_configs,
            pcc_check=0.995 if self.pcc_check else None,
        )
        self.conv2 = BOS_TTNN_Conv(
            self.image_shape,
            self.hidden_channels,
            self.out_channels,
            self.base_address + "cv2.",
            self.device,
            self.cv2_kernel,
            self.stride,
            self.padding,
            layer_configs=layer_configs,
            pcc_check=0.995 if self.pcc_check else None,
        )

        self.use_shortcut = use_shortcut and in_channels == out_channels

    def forward(self, x, out_memory_config=False):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.use_shortcut:
            if out_memory_config:
                output = ttnn.add(x, out, memory_config=out.memory_config())
            else:
                output = ttnn.add(x, out)
            if self.pcc_check is not None:
                torch_input_x = tt_to_torch_tensor(x)
                torch_input_out = tt_to_torch_tensor(out)
                golden_output = torch_input_x + torch_input_out
                torch_output = tt_to_torch_tensor(output)
                _, pcc = comp_pcc(golden_output, torch_output)
                print(
                    ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",
                    self.base_address,
                    "Bottleneck Add PCC =",
                    pcc,
                )
                assert pcc >= self.pcc_check
        else:
            output = out

        return output


class BOS_TTNN_CSP(nn.Module):
    def __init__(
        self,
        image_shape,
        in_channels,
        out_channels,
        depth,
        base_address,
        device,
        shortcut=False,
        add=True,
        layer_configs={},
        pcc_check=None,
    ):
        super().__init__()

        self.device = device
        self.image_shape = (
            image_shape if isinstance(image_shape, collections.abc.Iterable) else [image_shape, image_shape]
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = self.out_channels // 2
        self.depth = depth
        self.shortcut = shortcut
        self.base_address = base_address
        self.pcc_check = pcc_check

        self.cv1_in_channels = self.in_channels
        self.conv1_a = BOS_TTNN_Conv(
            self.image_shape,
            self.cv1_in_channels,
            self.hidden_channels,
            self.base_address + "cv1_a.",
            self.device,
            1,
            1,
            0,
            layer_configs=layer_configs,
            pcc_check=self.pcc_check,
        )
        self.conv1_b = BOS_TTNN_Conv(
            self.image_shape,
            self.cv1_in_channels,
            self.hidden_channels,
            self.base_address + "cv1_b.",
            self.device,
            1,
            1,
            0,
            layer_configs=layer_configs,
            pcc_check=self.pcc_check,
        )
        self.cv2_in_channels = (2 + self.depth) * self.hidden_channels
        self.cv2_out_channels = self.out_channels
        self.conv2 = BOS_TTNN_Conv(
            self.image_shape,
            self.cv2_in_channels,
            self.cv2_out_channels,
            self.base_address + "cv2.",
            self.device,
            1,
            1,
            0,
            layer_configs=layer_configs,
            pcc_check=self.pcc_check,
        )

        self.residual_layers = nn.ModuleList(
            BOS_TTNN_Bottleneck(
                self.image_shape,
                self.hidden_channels,
                self.hidden_channels,
                self.base_address + "m." + str(_) + ".",
                self.device,
                use_shortcut=self.shortcut,
                kernel=[3, 3],
                stride=1,
                padding=1,
                expansion=1,
                layer_configs=layer_configs,
                pcc_check=self.pcc_check,
            )
            for _ in range(self.depth)
        )

    def forward(self, x, pad_if_optimal=False):
        x1 = self.conv1_a(x)
        x2 = self.conv1_b(x)

        input_sharded_memory_config = x1.memory_config()

        outputs = [x1, x2]

        for i, residual_block in enumerate(self.residual_layers):
            x2 = residual_block(x2, i == self.depth - 1)
            outputs.append(x2)

        if pad_if_optimal:
            for i, tensor in enumerate(outputs):
                if tensor.shape[-2] % 32 != 0:
                    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
                    outputs[i] = ttnn.pad(
                        tensor, ((0, 0), (0, 0), (0, _nearest_32(tensor.shape[-2]) - tensor.shape[-2]), (0, 0)), 0
                    )
                    outputs[i] = ttnn.to_layout(outputs[i], ttnn.TILE_LAYOUT)

        # print(self.base_address)
        # for i in range(len(outputs)):
        #     print(outputs[i].shape, outputs[i].memory_config())
        # print()

        total_width = len(outputs) * self.hidden_channels
        input_sharded_spec = input_sharded_memory_config.shard_spec
        shard_height = input_sharded_spec.shape[0]
        input_sharded_memory_layout = str(input_sharded_memory_config.memory_layout)
        if "HEIGHT" in input_sharded_memory_layout:
            out_sharded_memory_config_strategy = ttnn.ShardStrategy.HEIGHT
        elif "BLOCK" in input_sharded_memory_layout:
            out_sharded_memory_config_strategy = ttnn.ShardStrategy.BLOCK
        elif "WIDTH" in input_sharded_memory_layout:
            out_sharded_memory_config_strategy = ttnn.ShardStrategy.WIDTH
        out_sharded_memory_config = ttnn.create_sharded_memory_config(
            (shard_height, total_width),
            core_grid=input_sharded_spec.grid,
            strategy=out_sharded_memory_config_strategy,
            use_height_and_width_as_shard_shape=True,
        )
        output = ttnn.concat(outputs, dim=3, memory_config=out_sharded_memory_config)

        final_output = self.conv2(output, interleave_if_optimal=True)

        return final_output


class BOS_TTNN_SPPF(nn.Module):
    def __init__(
        self,
        image_shape,
        in_channels,
        out_channels,
        pool_kernel,
        pool_stride,
        base_address,
        device,
        layer_configs={},
        pcc_check=None,
    ):
        super().__init__()

        self.device = device
        self.image_shape = (
            image_shape if isinstance(image_shape, collections.abc.Iterable) else [image_shape, image_shape]
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_kernel = (
            pool_kernel if isinstance(pool_kernel, collections.abc.Iterable) else (pool_kernel, pool_kernel)
        )
        self.pool_stride = (
            pool_stride if isinstance(pool_stride, collections.abc.Iterable) else (pool_stride, pool_stride)
        )
        self.base_address = base_address
        self.pcc_check = pcc_check

        self.conv1 = BOS_TTNN_Conv(
            self.image_shape,
            self.in_channels,
            self.in_channels // 2,
            self.base_address + "cv1.",
            self.device,
            1,
            1,
            0,
            layer_configs=layer_configs,
            pcc_check=self.pcc_check,
        )
        self.conv2 = BOS_TTNN_Conv(
            self.image_shape,
            self.in_channels * 2,
            self.out_channels,
            self.base_address + "cv2.",
            self.device,
            1,
            1,
            0,
            layer_configs=layer_configs,
            pcc_check=self.pcc_check,
        )

        self.pool = BOS_TTNN_MaxPool(self.device, self.base_address + ".m", pcc_check=self.pcc_check)

    def forward(self, x):
        x = self.conv1(x)

        y1 = self.pool(x, self.image_shape)

        y2 = self.pool(y1, self.image_shape)

        y3 = self.pool(y2, self.image_shape)

        outputs = [x, y1, y2, y3]

        reshard_mem_config = outputs[0].memory_config()
        for i in range(4):
            if outputs[i].shape[-2] % 32 != 0:
                outputs[i] = ttnn.reshard(outputs[i], reshard_mem_config)
                outputs[i] = ttnn.to_layout(outputs[i], ttnn.ROW_MAJOR_LAYOUT)
                outputs[i] = ttnn.pad(
                    outputs[i],
                    [(0, 0), (0, 0), (0, _nearest_32(outputs[i].shape[-2]) - outputs[i].shape[-2]), (0, 0)],
                    0,
                )
                outputs[i] = ttnn.to_layout(outputs[i], ttnn.TILE_LAYOUT)
            outputs[i] = ttnn.sharded_to_interleaved(outputs[i], ttnn.L1_MEMORY_CONFIG)

        output = ttnn.concat(outputs, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        output = self.conv2(output)

        return output
