# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.bos_model.ufld_v2.ttnn.common import TtnnUFLDV2Conv2D
from models.bos_model.ufld_v2.ttnn.ttnn_basic_block import TtnnBasicBlock


class TtnnResnet34:
    def __init__(self, conv_args, conv_pth, device, slice_config=ttnn.Conv2dL1FullSliceConfig, maxpool_dram=False):
        self.maxpool_args = conv_args.maxpool
        self.device = device
        self.maxpool_dram = maxpool_dram
        self.conv1 = TtnnUFLDV2Conv2D(
            conv_args.conv1,
            conv_pth.conv1,
            device=self.device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            dealloc_act=True,
            activation_dtype=ttnn.bfloat8_b,
            slice_config=slice_config,
            shard_layout=None,
            act_block_h_override=128,
        )
        self.layer1_0 = TtnnBasicBlock(
            conv_args.layer1[0],
            conv_pth.layer1_0,
            device=self.device,
            is_downsample=False,
            precision=ttnn.bfloat8_b,
            slice_config=slice_config,
        )
        self.layer1_1 = TtnnBasicBlock(
            conv_args.layer1[1],
            conv_pth.layer1_1,
            device=self.device,
            is_downsample=False,
            precision=ttnn.bfloat8_b,
            slice_config=slice_config,
        )
        self.layer1_2 = TtnnBasicBlock(
            conv_args.layer1[2],
            conv_pth.layer1_2,
            device=self.device,
            is_downsample=False,
            precision=ttnn.bfloat8_b,
            slice_config=slice_config,
        )
        self.layer2_0 = TtnnBasicBlock(
            conv_args.layer2[0],
            conv_pth.layer2_0,
            device=self.device,
            is_downsample=True,
            precision=ttnn.bfloat8_b,
            slice_config=slice_config,
        )
        self.layer2_1 = TtnnBasicBlock(
            conv_args.layer2[1],
            conv_pth.layer2_1,
            device=self.device,
            is_downsample=False,
            precision=ttnn.bfloat8_b,
            slice_config=slice_config,
        )
        self.layer2_2 = TtnnBasicBlock(
            conv_args.layer2[2],
            conv_pth.layer2_2,
            device=self.device,
            is_downsample=False,
            precision=ttnn.bfloat8_b,
            slice_config=slice_config,
        )
        self.layer2_3 = TtnnBasicBlock(
            conv_args.layer2[3],
            conv_pth.layer2_3,
            device=self.device,
            is_downsample=False,
            precision=ttnn.bfloat8_b,
            slice_config=slice_config,
        )
        self.layer3_0 = TtnnBasicBlock(
            conv_args.layer3[0],
            conv_pth.layer3_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            slice_config=slice_config,
        )
        self.layer3_1 = TtnnBasicBlock(
            conv_args.layer3[1],
            conv_pth.layer3_1,
            device=self.device,
            is_downsample=False,
            blk_sharded=True,
            slice_config=slice_config,
        )
        self.layer3_2 = TtnnBasicBlock(
            conv_args.layer3[2],
            conv_pth.layer3_2,
            device=self.device,
            is_downsample=False,
            blk_sharded=True,
            slice_config=slice_config,
        )
        self.layer3_3 = TtnnBasicBlock(
            conv_args.layer3[3],
            conv_pth.layer3_3,
            device=self.device,
            is_downsample=False,
            blk_sharded=True,
            slice_config=slice_config,
        )
        self.layer3_4 = TtnnBasicBlock(
            conv_args.layer3[4],
            conv_pth.layer3_4,
            device=self.device,
            is_downsample=False,
            blk_sharded=True,
            slice_config=slice_config,
        )
        self.layer3_5 = TtnnBasicBlock(
            conv_args.layer3[5],
            conv_pth.layer3_5,
            device=self.device,
            is_downsample=False,
            blk_sharded=True,
            slice_config=slice_config,
        )
        self.layer4_0 = TtnnBasicBlock(
            conv_args.layer4[0],
            conv_pth.layer4_0,
            device=self.device,
            is_downsample=True,
            blk_sharded=True,
            slice_config=slice_config,
        )
        self.layer4_1 = TtnnBasicBlock(
            conv_args.layer4[1],
            conv_pth.layer4_1,
            device=self.device,
            is_downsample=False,
            blk_sharded=True,
            slice_config=slice_config,
        )
        self.layer4_2 = TtnnBasicBlock(
            conv_args.layer4[2],
            conv_pth.layer4_2,
            device=self.device,
            is_downsample=False,
            blk_sharded=True,
            slice_config=slice_config,
        )

    def __call__(self, input, batch_size=1, min_channels=16, shard_height_for_maxcores=16000):  # for 5x4 core
        n, c, h, w = input.shape
        channel_padding_needed = min_channels - c
        x = ttnn.pad(input, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)

        x = ttnn.permute(x, (0, 2, 3, 1))
        x = ttnn.reshape(x, (1, 1, n * h * w, min_channels))
        x1, out_ht, out_wdth = self.conv1(x)
        ttnn.deallocate(x)

        if self.maxpool_dram:
            x1 = ttnn.max_pool2d(
                x1,
                batch_size=batch_size,
                input_h=out_ht,
                input_w=out_wdth,
                channels=x1.shape[-1],
                kernel_size=[self.maxpool_args.kernel_size, self.maxpool_args.kernel_size],
                stride=[self.maxpool_args.stride, self.maxpool_args.stride],
                padding=[self.maxpool_args.padding, self.maxpool_args.padding],
                dilation=[self.maxpool_args.dilation, self.maxpool_args.dilation],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            x1 = ttnn.max_pool2d(
                x1,
                batch_size=batch_size,
                input_h=out_ht,
                input_w=out_wdth,
                channels=x1.shape[-1],
                kernel_size=[self.maxpool_args.kernel_size, self.maxpool_args.kernel_size],
                stride=[self.maxpool_args.stride, self.maxpool_args.stride],
                padding=[self.maxpool_args.padding, self.maxpool_args.padding],
                dilation=[self.maxpool_args.dilation, self.maxpool_args.dilation],
            )

        # === Directly convert to tile layout to avoid extra copy ===#
        x = ttnn.to_layout(x1, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x1)

        x = self.layer1_0(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)

        return x
