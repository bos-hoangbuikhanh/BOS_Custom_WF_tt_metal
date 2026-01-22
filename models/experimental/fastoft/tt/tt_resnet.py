# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from .common import GroupNorm, GroupNormDRAM
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d, MaxPool2dConfiguration

from loguru import logger

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TTBasicBlock:
    expansion = 1
    block_counter = 0

    def __init__(self, device, state_dict, layer_args):
        self.block_id = TTBasicBlock.block_counter
        TTBasicBlock.block_counter += 1
        self.conv1 = TtConv2d(layer_args.conv1["optimized_configuration"], device)
        self.conv2 = TtConv2d(layer_args.conv2["optimized_configuration"], device)

        if "downsample" in state_dict.keys():
            self.downsample = True
            self.downsample_conv = TtConv2d(layer_args.downsample[0]["optimized_configuration"], device)
        else:
            self.downsample = None

    def forward(self, device, x, num_splits=1, move_to_dram=False):
        if use_signpost:
            signpost(header=f"TTBasicBlock {self.block_id} forward started")
        identity = x
        out = self.conv1(x)
        logger.debug(
            f"FORWARD X Input shape: {x.shape}, dtype: {x.dtype}, layout: {x.layout} memory_config: {x.memory_config()}"
        )

        out = self.conv2(out)
        logger.debug(f"Conv2 output shape: {out.shape}")
        logger.debug(f"BN2 output shape: {out.shape}")

        if self.downsample is not None:
            identity = self.downsample_conv(identity)
        else:
            logger.debug(f"reshape x shape: {x.shape} self.downsample: {self.downsample}")
            # x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))

        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)

        ttnn.add_(out, identity, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)])
        
        if move_to_dram:
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            identity = ttnn.to_memory_config(identity, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        if use_signpost:
            signpost(header=f"TTBasicBlock {self.block_id} forward finished")
        return out


class TTResNetFeatures:
    def __init__(self, device, parameters, conv_pt, block, layers, return_intermediates=False, num_of_layers=4, preprocess_conv1=False):
        self.conv1 = TtConv2d(conv_pt.conv1["optimized_configuration"], device)
        self.maxpool = TtMaxPool2d(
            configuration=MaxPool2dConfiguration(
                input_height=conv_pt.maxpool.input_height,
                input_width=conv_pt.maxpool.input_width,
                channels=conv_pt.maxpool.input_channels,
                batch_size=conv_pt.maxpool.batch_size,
                kernel_size=(conv_pt.maxpool.kernel_size, conv_pt.maxpool.kernel_size),
                stride=(conv_pt.maxpool.stride, conv_pt.maxpool.stride),
                padding=(conv_pt.maxpool.padding, conv_pt.maxpool.padding),
                dilation=(conv_pt.maxpool.dilation, conv_pt.maxpool.dilation),
                deallocate_input=True,
                output_layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
            ),
            device=device,
        )

        assert num_of_layers in [2,3,4], "num_of_layers must be 2, 3, or 4"

        self.layer1 = self._make_layer(device, parameters.layer1, conv_pt.layer1, block, layers[0])
        self.layer2 = self._make_layer(device, parameters.layer2, conv_pt.layer2, block, layers[1])
        self.layer3 = self._make_layer(device, parameters.layer3, conv_pt.layer3, block, layers[2]) if num_of_layers >= 3 else None
        self.layer4 = self._make_layer(
            device,
            parameters.layer4,
            conv_pt.layer4,
            block,
            layers[3],
        ) if num_of_layers >= 4 else None
        self.return_intermediates = return_intermediates
        self.preprocess_conv1 = preprocess_conv1

        self.num_splits_gn = 2  # Number of splits for GroupNorm to fit into L1
        self.num_slices = 2  # Number of slices used for partial sharding during concatenation of GN outputs

    def _make_layer(self, device, parameters, conv_pt, block, blocks):
        layers = []
        layers.append(
            block(
                device,
                parameters[0],
                conv_pt[0],
            )
        )
        for i in range(1, blocks):
            layers.append(
                block(
                    device,
                    parameters[i],
                    conv_pt[i],
                )
            )
        return layers

    def _run_layer(self, device, x, layer, return_intermediates=False):
        """Run a layer with optional intermediate activation capture."""
        intermediates = []

        if return_intermediates:
            intermediates.append(ttnn.to_torch(x).permute(0, 3, 1, 2))

        for block in layer:
            x = block.forward(device, x)

            if return_intermediates:
                # Clone/copy each intermediate activation
                intermediates.append(ttnn.to_torch(x).permute(0, 3, 1, 2))

        if return_intermediates:
            return x, intermediates
        else:
            return x, None

    def forward(self, device, x):
        if use_signpost:
            signpost(header="ResNet module started")

        if not self.preprocess_conv1:
            # conv1 is moved to preprocessing step; to enable pipelining
            x = self.conv1(x)

        host_x = ttnn.to_torch(x).permute(0, 3, 1, 2) if self.return_intermediates else None
        conv_1 = self.maxpool(x)
        host_mp = ttnn.to_torch(conv_1).permute(0, 3, 1, 2) if self.return_intermediates else None

        feats4, i4 = self._run_layer(device, conv_1, self.layer1, return_intermediates=self.return_intermediates)

        ttnn.deallocate(conv_1)
        feats8, i8 = self._run_layer(device, feats4, self.layer2, return_intermediates=self.return_intermediates)
        feats8_interleaved = ttnn.sharded_to_interleaved(feats8, ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(feats4)
        feats16, i16 = self._run_layer(device, feats8, self.layer3, return_intermediates=self.return_intermediates) if self.layer3 is not None else (None, None)
        feats16_interleaved = ttnn.sharded_to_interleaved(feats16, ttnn.DRAM_MEMORY_CONFIG) if feats16 is not None else None
        ttnn.deallocate(feats8)

        feats32, i32 = self._run_layer(device, feats16, self.layer4, return_intermediates=self.return_intermediates) if self.layer4 is not None else (None, None)
        feats32_interleaved = ttnn.sharded_to_interleaved(feats32, ttnn.DRAM_MEMORY_CONFIG) if feats32 is not None else None
        if feats16 is not None:
            ttnn.deallocate(feats16)

        if use_signpost:
            signpost(header="ResNet module finished")

        if self.return_intermediates:
            return (
                [host_x, i4, i8, i16, i32, host_mp],
                feats8_interleaved,
                feats16_interleaved,
                feats32_interleaved,
            )
        else:
            return feats8_interleaved, feats16_interleaved, feats32_interleaved
