# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

# from models.bos_model.oft.tt.common import Conv
from models.tt_cnn.tt.builder import TtConv2d
from models.bos_model.oft.tt.common import GroupNorm
from models.bos_model.oft.tt.tt_resnet import TTResNetFeatures
from models.bos_model.oft.tt.tt_oft import OFT as TtOFT
from loguru import logger

try:
    from tracy import signpost

except ModuleNotFoundError:

    def signpost(*args, **kwargs):
        pass


class TTOftNet:
    def __init__(
        self,
        device,
        state_dict,
        layer_args,
        block,
        layers,
        mean,
        std,
        input_shape_hw,
        calib,
        grid,
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4,
    ):
        self.frontend = TTResNetFeatures(device, state_dict.frontend, layer_args.frontend, block, layers)
        self.lat8 = TtConv2d(layer_args.lat8["optimized_configuration"], device)
        self.bn8 = GroupNorm(state_dict.bn8, layer_args.bn8)

        self.lat16 = TtConv2d(layer_args.lat16["optimized_configuration"], device)
        self.bn16 = GroupNorm(state_dict.bn16, layer_args.bn16)

        self.lat32 = TtConv2d(layer_args.lat32["optimized_configuration"], device)
        self.bn32 = GroupNorm(state_dict.bn32, layer_args.bn32)

        self.oft8 = TtOFT(
            device,
            state_dict.oft8,
            256,
            grid_res,
            grid_height,
            [int(x * 1 / 8) for x in input_shape_hw],
            calib,
            grid,
            scale=1 / 8,
            use_precomputed_grid=True,
            num_slices=18,
        )
        self.oft16 = TtOFT(
            device,
            state_dict.oft16,
            256,
            grid_res,
            grid_height,
            [int(x * 1 / 16) for x in input_shape_hw],
            calib,
            grid,
            scale=1 / 16,
            use_precomputed_grid=True,
            num_slices=12,
        )
        self.oft32 = TtOFT(
            device,
            state_dict.oft32,
            256,
            grid_res,
            grid_height,
            [int(x * 1 / 32) for x in input_shape_hw],
            calib,
            grid,
            scale=1 / 32,
            use_precomputed_grid=True,
            num_slices=11,
        )
        ttnn.device.ReadDeviceProfiler(device)
        self.topdown = [
            block(
                device,
                state_dict.topdown[i],
                state_dict.layer_args.topdown[i],
                is_sliced=True,
            )
            for i in range(topdown_layers)
        ]

        self.head = TtConv2d(layer_args.head["optimized_configuration"], device)
        self.mean = ttnn.from_torch(
            mean, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.std = ttnn.from_torch(
            std, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def forward_normalization(self, device, input_tensor):
        """Normalize input tensor by mean and std-dev"""
        # Normalize by mean and std-dev
        input_tensor = input_tensor - self.mean
        input_tensor = ttnn.divide(input_tensor, self.std)
        return input_tensor

    def forward_lateral_layers(self, device, feats8, feats16, feats32):
        """Apply lateral layers to convert image features to common feature size"""
        # Apply lateral layers to convert image features to common feature size
        lat8 = self.lat8(feats8)
        if lat8.layout == ttnn.TILE_LAYOUT:
            lat8 = ttnn.to_layout(lat8, ttnn.ROW_MAJOR_LAYOUT)
        lat8 = self.bn8(device, lat8, shard="HS")
        lat8 = ttnn.relu(lat8)
        lat8 = ttnn.sharded_to_interleaved(lat8, ttnn.DRAM_MEMORY_CONFIG)
        lat8 = ttnn.to_layout(lat8, ttnn.TILE_LAYOUT)

        lat16 = self.lat16(feats16)
        if lat16.layout == ttnn.TILE_LAYOUT:
            lat16 = ttnn.to_layout(lat16, ttnn.ROW_MAJOR_LAYOUT)
        lat16 = self.bn16(device, lat16, shard="HS")
        lat16 = ttnn.relu(lat16)
        lat16 = ttnn.sharded_to_interleaved(lat16, ttnn.DRAM_MEMORY_CONFIG)
        lat16 = ttnn.to_layout(lat16, ttnn.TILE_LAYOUT)

        lat32 = self.lat32(feats32)
        if lat32.layout == ttnn.TILE_LAYOUT:
            lat32 = ttnn.to_layout(lat32, ttnn.ROW_MAJOR_LAYOUT)
        lat32 = self.bn32(device, lat32, shard="BS")
        lat32 = ttnn.relu(lat32)
        lat32 = ttnn.sharded_to_interleaved(lat32, ttnn.DRAM_MEMORY_CONFIG)
        lat32 = ttnn.to_layout(lat32, ttnn.TILE_LAYOUT)

        return lat8, lat16, lat32

    def forward_oft(self, device, lat8, lat16, lat32, calib, grid, return_intermediates=False):
        """Apply orthographic feature transform (OFT) and sum the results"""
        signpost(header="Oft module")

        # Apply OFT and sum
        ortho8, integral_img8, bbox_top_left8, bbox_btm_right8, bbox_top_right8, bbox_btm_left8 = self.oft8.forward(
            device, lat8, calib, grid
        )  # ortho8
        # ttnn.deallocate(lat8)
        lat8 = ttnn.to_memory_config(lat8, ttnn.DRAM_MEMORY_CONFIG)
        (
            ortho16,
            integral_img16,
            bbox_top_left16,
            bbox_btm_right16,
            bbox_top_right16,
            bbox_btm_left16,
        ) = self.oft16.forward(device, lat16, calib, grid)
        # ttnn.deallocate(lat16)
        lat16 = ttnn.to_memory_config(lat16, ttnn.DRAM_MEMORY_CONFIG)
        (
            ortho32,
            integral_img32,
            bbox_top_left32,
            bbox_btm_right32,
            bbox_top_right32,
            bbox_btm_left32,
        ) = self.oft32.forward(device, lat32, calib, grid)
        # ttnn.deallocate(lat32)
        lat32 = ttnn.to_memory_config(lat32, ttnn.DRAM_MEMORY_CONFIG)

        ortho = ortho8 + ortho16 + ortho32

        signpost(header="Oft module finished")
        logger.debug(f"Ortho shape: {ortho.shape}, dtype: {ortho.dtype}")

        if ortho.layout == ttnn.TILE_LAYOUT:
            ortho = ttnn.to_layout(ortho, ttnn.ROW_MAJOR_LAYOUT)
        logger.debug(
            f"Ortho shape: {ortho.shape}, dtype: {ortho.dtype} layout: {ortho.layout} memory_config: {ortho.memory_config()}"
        )

        if return_intermediates:
            return (
                lat8,
                lat16,
                lat32,
                integral_img8,
                integral_img16,
                integral_img32,
                bbox_top_left8,
                bbox_btm_right8,
                bbox_top_right8,
                bbox_btm_left8,
                bbox_top_left16,
                bbox_btm_right16,
                bbox_top_right16,
                bbox_btm_left16,
                bbox_top_left32,
                bbox_btm_right32,
                bbox_top_right32,
                bbox_btm_left32,
                ortho8,
                ortho16,
                ortho32,
            ), ortho
        else:
            return None, ortho

    def forward_topdown_network(self, device, ortho):
        """Apply topdown network"""
        signpost(header="Topdown started")
        td = ortho
        for layer in self.topdown:
            logger.debug(f"Topdown layer {layer=}")
            td = layer.forward(device, td, gn_shard="HS", num_splits=2)  # hangs on top down with these settings;
            # td = layer.forward(device, td)
        signpost(header="Topdown finished")
        return td

    def forward_predict_encoded_outputs(self, device, td):
        """Predict encoded outputs and slice them"""
        signpost(header="Head started")
        out_h, out_w = 159, 159  # todo plumb return output shape from common conv wrapper
        outputs = self.head(td)
        logger.debug(f"Head output shape: {outputs.shape}, dtype: {outputs.dtype} {out_h=} {out_w=}")
        outputs = ttnn.permute(outputs, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        outputs = ttnn.reshape(outputs, (1, -1, 9, out_h, out_w))
        signpost(header="Head finished")
        signpost(header="Slicing started")
        slices = [1, 3, 3, 2]
        start = 0
        parts = []
        for i in range(len(slices)):
            parts.append(outputs[:, :, start : start + slices[i], :, :])
            start += slices[i]
        parts[0] = ttnn.squeeze(parts[0], dim=2)  # remove the 1 slice dimension
        for part in parts:
            logger.debug(f"Part shape: {part.shape}, dtype: {part.dtype}")
        signpost(header="Slicing finished")
        return parts

    def forward(self, device, input_tensor, calib, grid, return_intermediates=False):
        signpost(header="OftNet module started")

        # Normalize input tensor
        normalized_input = self.forward_normalization(device, input_tensor)

        # Run frontend network
        feats8, feats16, feats32 = self.frontend.forward(device, normalized_input)

        ttnn.device.ReadDeviceProfiler(device)
        # Apply lateral layers
        lat8, lat16, lat32 = self.forward_lateral_layers(device, feats8, feats16, feats32)

        # Apply OFT transformation
        oft_intermediates, ortho = self.forward_oft(
            device, lat8, lat16, lat32, calib, grid, return_intermediates=return_intermediates
        )

        # Apply topdown network
        td = self.forward_topdown_network(device, ortho)

        # Predict encoded outputs
        tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = self.forward_predict_encoded_outputs(device, td)

        signpost(header="OftNet finished")

        if return_intermediates:
            import torch

            (
                lat8,
                lat16,
                lat32,
                integral_img8,
                integral_img16,
                integral_img32,
                bbox_top_left8,
                bbox_btm_right8,
                bbox_top_right8,
                bbox_btm_left8,
                bbox_top_left16,
                bbox_btm_right16,
                bbox_top_right16,
                bbox_btm_left16,
                bbox_top_left32,
                bbox_btm_right32,
                bbox_top_right32,
                bbox_btm_left32,
                ortho8,
                ortho16,
                ortho32,
            ) = oft_intermediates
            return (
                [
                    (
                        normalized_input,
                        feats8,
                        feats16,
                        feats32,
                        lat8,
                        lat16,
                        lat32,
                        integral_img8,
                        integral_img16,
                        integral_img32,
                        torch.cat([ttnn.to_torch(bbox_top_left8[i]) for i in range(len(bbox_top_left8))], dim=2),
                        torch.cat([ttnn.to_torch(bbox_btm_right8[i]) for i in range(len(bbox_btm_right8))], dim=2),
                        torch.cat([ttnn.to_torch(bbox_top_right8[i]) for i in range(len(bbox_top_right8))], dim=2),
                        torch.cat([ttnn.to_torch(bbox_btm_left8[i]) for i in range(len(bbox_btm_left8))], dim=2),
                        torch.cat([ttnn.to_torch(bbox_top_left16[i]) for i in range(len(bbox_top_left16))], dim=2),
                        torch.cat([ttnn.to_torch(bbox_btm_right16[i]) for i in range(len(bbox_btm_right16))], dim=2),
                        torch.cat([ttnn.to_torch(bbox_top_right16[i]) for i in range(len(bbox_top_right16))], dim=2),
                        torch.cat([ttnn.to_torch(bbox_btm_left16[i]) for i in range(len(bbox_btm_left16))], dim=2),
                        torch.cat([ttnn.to_torch(bbox_top_left32[i]) for i in range(len(bbox_top_left32))], dim=2),
                        torch.cat([ttnn.to_torch(bbox_btm_right32[i]) for i in range(len(bbox_btm_right32))], dim=2),
                        torch.cat([ttnn.to_torch(bbox_top_right32[i]) for i in range(len(bbox_top_right32))], dim=2),
                        torch.cat([ttnn.to_torch(bbox_btm_left32[i]) for i in range(len(bbox_btm_left32))], dim=2),
                        ortho8,
                        ortho16,
                        ortho32,
                        ortho,
                        ttnn.to_torch(calib, dtype=torch.float32),
                        ttnn.to_torch(grid, dtype=torch.float32),
                        td,
                    ),
                    (
                        "image",
                        "feats8",
                        "feats16",
                        "feats32",
                        "lat8",
                        "lat16",
                        "lat32",
                        "integral_img8",
                        "integral_img16",
                        "integral_img32",
                        "bbox_top_left8",
                        "bbox_btm_right8",
                        "bbox_top_right8",
                        "bbox_btm_left8",
                        "bbox_top_left16",
                        "bbox_btm_right16",
                        "bbox_top_right16",
                        "bbox_btm_left16",
                        "bbox_top_left32",
                        "bbox_btm_right32",
                        "bbox_top_right32",
                        "bbox_btm_left32",
                        "ortho8",
                        "ortho16",
                        "ortho32",
                        "ortho",
                        "calib",
                        "grid",
                        "td",
                    ),
                ],
                tt_scores,
                tt_pos_offsets,
                tt_dim_offsets,
                tt_ang_offsets,
            )
        else:
            return [(), ()], tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets
