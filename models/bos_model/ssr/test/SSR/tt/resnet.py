import os
from test.SSR.tt.utils.misc import setup_l1_sharded_config
from typing import Optional, Type, Union

import torch
from bos_metal import ttnn
from bos_metal.core import BaseModule
from bos_metal.operations import Clone, Functional, Sequential
from bos_metal.operations.conv import Conv2d
from bos_metal.operations.pool import MaxPool2d, Pool2dConfig
from mmdet.models.builder import BACKBONES


class TTBasicBlock(BaseModule):
    expansion = 1
    count = 0

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        debug=False,
        **kwargs,
    ):
        super(TTBasicBlock, self).__init__()
        self.stride = stride
        self.dilation = dilation
        self.debug = debug

        self.conv1 = Conv2d(
            inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=True, requires_shape=False
        )
        self.conv2 = Conv2d(
            planes,
            planes,
            3,
            padding=1,
            bias=True,
            # activation='relu',
            requires_shape=False,
        )
        self.downsample = downsample
        self.relu = ttnn.relu
        self.add = Add(deallocate_input=True, requires_shape=False)

    # TODO: config for Conv2dSplit
    def forward(self, x):
        """Forward function."""
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)
        out = ttnn.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        self.add(out, identity)
        out = self.relu(out)

        return out


class Add(BaseModule):
    def __init__(self, inplace=True):
        super().__init__()
        if inplace:
            self.add = ttnn.add_  # type: ignore
        else:
            self.add = ttnn.add  # type: ignore

    def forward(self, a, b, activations=None):
        # Process input a
        a = ttnn.to_layout(a, layout=ttnn.Layout.TILE) if a.layout != ttnn.Layout.TILE else a

        # Process input b
        b = b.reshape(a.shape) if b.shape != a.shape else b
        b = ttnn.to_layout(b, layout=ttnn.Layout.TILE) if b.layout != ttnn.Layout.TILE else b

        # Perform addition
        a = self.add(a, b, activations=activations)

        # Deallocate identity
        ttnn.deallocate(b)
        return a


def conv3x3_tt(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, activation=None
) -> Conv2d:
    """3x3 convolution with padding"""
    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=(dilation, dilation),
        groups=groups,
        bias=True,
        dilation=(dilation, dilation),
        activation=activation,
    )


def conv1x1_tt(in_planes: int, out_planes: int, stride: int = 1, activation=None) -> Conv2d:
    """1x1 convolution"""
    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=(1, 1),
        stride=(stride, stride),
        groups=1,
        bias=True,
        dilation=(1, 1),
        activation=activation,
    )


class TTBottleneck(BaseModule):
    """From torchvision.models.resnet.Bottleneck"""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        downsample: Optional[Sequential] = None,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1_tt(inplanes, width, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU))
        self.conv2 = conv3x3_tt(
            width, width, stride, groups, dilation, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        )
        self.conv3 = conv1x1_tt(width, planes * self.expansion, activation=None)
        self.downsample = downsample
        self.relu = Functional(ttnn.relu)  # type: ignore
        self.add = Add()
        self.clone = Clone()

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        return self.add(identity, out, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)])


class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (BaseModule): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(
        self,
        block: Union[TTBasicBlock, TTBottleneck],
        inplanes,
        planes,
        num_blocks,
        stride=1,
        dilation=1,
        *,
        style="pytorch",
        downsample_first=True,
        dcn=None,
        debug=False,
        **kwargs,
    ):
        self.block = block
        self.dcn = dcn
        self.downsample_first = downsample_first
        self.debug = debug
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            downsample.extend(
                [
                    Conv2d(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias=True,
                        activation=None,
                    ),
                ]
            )
            downsample = Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    downsample=downsample,
                    stride=stride,
                    dilation=dilation,
                )
            )
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                    )
                )

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                    )
                )
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                )
            )
        super(ResLayer, self).__init__(*layers)


@BACKBONES.register_module("ResNet_tt")
class ResNet(BaseModule):
    """ResNet backbone."""

    arch_settings = {
        18: (TTBasicBlock, (2, 2, 2, 2)),
        34: (TTBasicBlock, (3, 4, 6, 3)),
        50: (TTBottleneck, (3, 4, 6, 3)),
        101: (TTBottleneck, (3, 4, 23, 3)),
        152: (TTBottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        avg_down=False,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        debug=False,
        **kwargs,
    ):
        super(ResNet, self).__init__()
        # Assertions
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")
        stem_channels = stem_channels or base_channels

        # Set attributes
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.debug = debug
        assert num_stages >= 1 and num_stages <= 4
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) < num_stages
        self.style = style
        self.avg_down = avg_down
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self.first_run = True
        self.slice_size = 192 * 320

        # Variable for persistent l1 memory
        self.sharded_l1_mem_config = None
        self.persistent_l1_spec = None

        # CONV1
        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=self.stem_channels,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=True,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )

        # CONV2_X
        self.maxpool = MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(1, 1),
            config=Pool2dConfig(
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            ),
        )

        # Res layers
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            planes = base_channels * 2**i
            res_layer = ResLayer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                dcn=dcn,
                debug=debug,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

    def _make_layer(
        self,
        block: Type[TTBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=(1, 1),
                    stride=(stride, stride),
                    bias=True,
                    activation=None,
                ),
            )

        layers = []
        layers.append(block(self.inplanes, planes, downsample, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return Sequential(*layers)

    def reset_batch_size(self, batch_size: int, modules: list[str] = None):
        if not modules:
            return

        def get_new_shape(child, batch_size):
            shape = list(child.input_shape)
            shape[0] = batch_size
            return tuple(shape)

        for name, child in self.named_modules():
            if isinstance(child, Conv2d) and any(name.startswith(layer) for layer in modules):
                new_shape = get_new_shape(child, batch_size)
                child.set_shapes(input_shape=new_shape)

    def reallocate_weights_and_bias(self):
        """Reallocate weights and bias for all Conv2d layers."""
        for name, child in self.named_modules():
            if isinstance(child, Conv2d):
                child.reallocate_weights_and_bias()

    def prepare_persistent_l1_spec(self, tensor):
        # Memory config
        self.sharded_l1_mem_config = setup_l1_sharded_config(tensor, device=self.device)
        temp_tensor = ttnn.to_memory_config(
            tensor,
            self.sharded_l1_mem_config,
        )
        # Reallocate to avoid fragmentation
        temp_tensor = ttnn.reallocate(temp_tensor)

        # Record spec
        self.persistent_l1_spec = temp_tensor.spec

        # Deallocate temp tensor
        ttnn.deallocate(temp_tensor)

    def forward(self, x):
        """Forward function."""
        if self.first_run:
            # Original tensor is splitted into 6 sub-tensor corresponding to 6 images.
            # Then, those images are sequentially forwarded through the network until layer3.0 (After finish layer3.0).
            self.reset_batch_size(batch_size=1, modules=["conv1", "layer1", "layer2", "layer3.0"])
            # After concatenating two sub-tensors into a single tensor, it will be forwarded through the network until the end of the model.
            self.reset_batch_size(
                batch_size=2, modules=["layer3.1", "layer3.2", "layer3.3", "layer3.4", "layer3.5", "layer4"]
            )

        # Prepare a spec of a persistent l1 memory (For trace mode)
        if self.first_run:
            self.prepare_persistent_l1_spec(x[0])

        # Forward
        temp_tensor = None
        # Forward 6 sub-tensors sequentially
        for i in range(len(x)):
            x_i = ttnn.allocate_tensor_on_device(self.persistent_l1_spec, self.device)
            x_i = ttnn.bos_reshard(x[i], x_i.memory_config(), x_i)

            # Conv1
            x_i = self.conv1(x_i)
            if self.debug:
                self.info("Resnet conv1")

            # Maxpool
            x_i = self.maxpool(x_i)
            if self.debug:
                self.info("Resnet maxpool")

            # ResNet layer 1, 2
            for j in range(2):
                res_layer = getattr(self, f"layer{j + 1}")
                if self.debug:
                    self.info(f"Running res_layer {j + 1}")
                x_i = res_layer(x_i)

            # ResNet layer3.0
            res_layer = getattr(self, f"layer3")
            x_i = res_layer[0](x_i)

            # Concat 2 sub-tensors into a single tensor then forward until the end of the model
            if i % 2 == 0:
                temp_tensor = x_i
            else:
                temp_tensor = (
                    ttnn.to_memory_config(temp_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
                    if temp_tensor.memory_config() != ttnn.L1_MEMORY_CONFIG
                    else temp_tensor
                )
                x_i = (
                    ttnn.to_memory_config(x_i, memory_config=ttnn.L1_MEMORY_CONFIG)
                    if x_i.memory_config() != ttnn.L1_MEMORY_CONFIG
                    else x_i
                )
                x_i = ttnn.concat((temp_tensor, x_i), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(temp_tensor)

                # ResNet layer 3.1 - 3.5
                res_layer = getattr(self, f"layer3")
                for j in range(1, 6):
                    x_i = res_layer[j](x_i)

                # ResNet layer 4
                res_layer = getattr(self, f"layer4")
                x_i = res_layer(x_i)

                # Concatenate them into a final output tensor
                x_i = ttnn.to_memory_config(x_i, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if i == 1:
                    out = x_i
                else:
                    out = ttnn.concat((out, x_i), dim=2)
                    ttnn.deallocate(x_i)

        # On the first run, reallocate Conv2d weights and biases in DRAM to reduce fragmentation
        if self.first_run:
            self.first_run = False
            self.reallocate_weights_and_bias()
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
            ttnn.ReadDeviceProfiler(self.device)
        return (out,)
