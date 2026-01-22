from typing import Tuple, Union

import torch
import torch.nn as nn

from bos_metal import ttnn
from bos_metal.core import BaseModule
from bos_metal.operations.conv import Conv2d
from bos_metal.operations.pool import MaxPool2d
from bos_metal.operations.binary import Add

from mmdet.models.builder import NECKS

class FPNConvModule(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 act='',
                 bias: Union[bool, str] = 'auto',
                 padding_mode: str = 'zeros',
                 cfg=None
                 ):
        super().__init__()
        official_padding_mode = ['zeros', 'circular']
        self.with_explicit_padding = padding_mode not in official_padding_mode

        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = True #not self.with_norm

        if self.with_explicit_padding:
            self.padding = padding
            
        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            activation=act,
            config=cfg,
        )
        # self.conv.conv.split_config.override_config(
        #     threshold=2227200
        # )
        
    def forward(self, x):
        x = self.conv(x)
        return x

@NECKS.register_module("FPN_tt")
class FPN(BaseModule):
    r"""Feature Pyramid Network.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 upsample_cfg:dict=dict(mode='nearest'),
                 ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.upsample_cfg = upsample_cfg.copy()
        self.conv_cfg = conv_cfg

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = FPNConvModule(
                in_channels[i],
                out_channels,
                1,
                bias=True,
                act='',
                cfg=conv_cfg
            )
            fpn_conv = FPNConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=True,
                act='',
                cfg=conv_cfg
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = FPNConvModule(
                    in_channels,
                    out_channels,
                    3,
                    bias=True,
                    stride=2,
                    padding=1,
                    act='',
                    cfg=conv_cfg
                )
                self.fpn_convs.append(extra_fpn_conv)
        
        self.adder = Add(deallocate_input=True, requires_shape=False)
    
    def upsample(self, x, x_shape, scale_factor):
        ret = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        ret = ttnn.sharded_to_interleaved(ret, memory_config=ttnn.L1_MEMORY_CONFIG)
        ret = ttnn.reshape(ret, (x_shape[0], x_shape[2], x_shape[3], x_shape[1]))
        ret = ttnn.upsample(ret, scale_factor=scale_factor)
        ret = ttnn.reshape(ret, (1, 1, ret.shape[0] * ret.shape[1] * ret.shape[2], ret.shape[3]))
        return ret

    def fallback_upsample(self, x, curr_shape, prev_shape):
        # we need fallback because the desired shape is 57 but the given shape is 29
        #  => can not handle edge case like this
        B_curr, C_curr, H_curr, W_curr = curr_shape
        B_prev, C_prev, H_prev, W_prev = prev_shape
        ret = ttnn.to_torch(x, torch.float32)
        ret = torch.reshape(ret, (B_curr, H_curr, W_curr, C_curr))
        ret = torch.permute(ret, (0, 3, 1, 2))  # B, C, H, W
        ret = torch.nn.functional.interpolate(ret, size=(H_prev, W_prev), mode='nearest')
        ret = torch.permute(ret, (0, 2, 3, 1))
        ret = torch.reshape(ret, (1, 1, B_prev*H_prev*W_prev, C_prev))
        ret = ttnn.from_torch(ret, dtype=ttnn.bfloat16, device=self.device)
        return ret
    
    def reallocate_weights_and_bias(self):
        """Reallocate weights and bias for all Conv2d layers."""
        for name, child in self.named_modules():
            if isinstance(child, Conv2d):
                child.reallocate_weights_and_bias()

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        
        # build laterals
        laterals = [
            {'tensor': lateral_conv(inputs[i + self.start_level]), 'shape': lateral_conv.conv.output_shape}
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # upsampled = self.fallback_upsample(x=laterals[i]['tensor'], curr_shape=laterals[i]['shape'], prev_shape=laterals[i-1]['shape'])
            upsampled = self.upsample(laterals[i]['tensor'], x_shape=laterals[i]['shape'], scale_factor=2)
            laterals[i - 1]['tensor'] = self.adder(laterals[i - 1]['tensor'], upsampled)
            laterals[i - 1]['tensor'] = ttnn.to_layout(laterals[i - 1]['tensor'], ttnn.ROW_MAJOR_LAYOUT)
            laterals[i - 1]['tensor'] = ttnn.to_memory_config(laterals[i - 1]['tensor'], ttnn.DRAM_MEMORY_CONFIG)

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]['tensor']) for i in range(used_backbone_levels)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                self.max_pool = MaxPool2d(kernel_size=1, stride=2, padding=0, requires_shape=False)
                for i in range(self.num_outs - used_backbone_levels):
                    self.max_pool.set_shapes(outs[-1].size())
                    outs.append(self.max_pool(outs[-1]))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError

                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs[-1] = ttnn.to_layout(outs[-1], ttnn.TILE_LAYOUT)
                        outs.append(self.fpn_convs[i](ttnn.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return outs