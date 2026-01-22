# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

# from models.experimental.oft.tt.common import Conv
from models.tt_cnn.tt.builder import TtConv2d
from .common import GroupNorm
from .tt_resnet import TTResNetFeatures
from .tt_oft import OFT as TtOFT
from loguru import logger
from ..reference.oftnet import OftMode

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
        oft_mode=OftMode.ORIGINAL,
        return_intermediates=False,
        preprocess_frontend_conv1=False,
    ):
        self.return_intermediates = return_intermediates

        if oft_mode == OftMode.OFT32 or oft_mode == OftMode.ORIGINAL:
            num_of_layers = 4
        elif oft_mode == OftMode.OFT16:
            num_of_layers = 3
        elif oft_mode == OftMode.OFT8:
            num_of_layers = 2
        self.frontend = TTResNetFeatures(device, state_dict.frontend, layer_args.frontend, block, layers,
                                         num_of_layers=num_of_layers, return_intermediates=False, preprocess_conv1=preprocess_frontend_conv1)

        # Lateral layers convert resnet outputs to a common feature size
        self.lat8 = (
            TtConv2d(layer_args.lat8["optimized_configuration"], device)
            if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT8)
            else None
        )

        self.lat16 = (
            TtConv2d(layer_args.lat16["optimized_configuration"], device)
            if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT16)
            else None
        )

        self.lat32 = (
            TtConv2d(layer_args.lat32["optimized_configuration"], device)
            if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT32)
            else None
        )

        # Orthographic feature transforms
        input_channels = 256  # Channels from lateral layers
        output_channels = 128  # Updated from 256 to 128 as per user's change

        self.oft8 = (
            TtOFT(
                device,
                state_dict.oft8,
                input_channels,
                output_channels,
                grid_res,
                grid_height,
                [int(x * 1 / 8) for x in input_shape_hw],
                calib,
                grid,
                scale=1 / 8,
                use_precomputed_grid=True,
                num_slices=44,
                mode="nearest",
                align_corners=True,
                return_intermediates=return_intermediates
            )
            if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT8)
            else None
        )
        self.oft16 = (
            TtOFT(
                device,
                state_dict.oft16,
                input_channels,
                output_channels,
                grid_res,
                grid_height,
                [int(x * 1 / 16) for x in input_shape_hw],
                calib,
                grid,
                scale=1 / 16,
                use_precomputed_grid=True,
                num_slices=12,
                mode="nearest",
                align_corners=True,
                return_intermediates=return_intermediates
            )
            if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT16)
            else None
        )
        self.oft32 = (
            TtOFT(
                device,
                state_dict.oft32,
                input_channels,
                output_channels,
                grid_res,
                grid_height,
                [int(x * 1 / 32) for x in input_shape_hw],
                calib,
                grid,
                scale=1 / 32,
                use_precomputed_grid=True,
                num_slices=11,
                mode="nearest",
                align_corners=True,
                return_intermediates=return_intermediates
            )
            if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT32)
            else None
        )
        # ttnn.device.ReadDeviceProfiler(device)  # disabled because of tracy; add switch to control it later

        self.topdown = [
            block(
                device,
                state_dict.topdown[i],
                state_dict.layer_args.topdown[i],
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

        self.oft_mode = oft_mode

    def forward_normalization(self, device, input_tensor):
        """Normalize input tensor by mean and std-dev"""
        # Normalize by mean and std-dev
        input_tensor = input_tensor - self.mean
        input_tensor = ttnn.div(input_tensor, self.std)
        return input_tensor

    def forward_lateral_layers(self, device, feats8, feats16, feats32):
        """Apply lateral layers to convert image features to common feature size"""
        # Apply lateral layers to convert image features to common feature size
        if self.lat8 is not None:
            lat8 = self.lat8(feats8)
            lat8 = ttnn.sharded_to_interleaved(lat8, ttnn.DRAM_MEMORY_CONFIG)
            lat8 = ttnn.to_layout(lat8, ttnn.TILE_LAYOUT)
        else:
            lat8 = None

        if self.lat16 is not None:
            lat16 = self.lat16(feats16)
            lat16 = ttnn.sharded_to_interleaved(lat16, ttnn.DRAM_MEMORY_CONFIG)
            lat16 = ttnn.to_layout(lat16, ttnn.TILE_LAYOUT)
        else:
            lat16 = None

        if self.lat32 is not None:
            lat32 = self.lat32(feats32)
            lat32 = ttnn.sharded_to_interleaved(lat32, ttnn.DRAM_MEMORY_CONFIG)
            lat32 = ttnn.to_layout(lat32, ttnn.TILE_LAYOUT)
        else:
            lat32 = None

        return lat8, lat16, lat32

    def forward_oft(self, device, lat8, lat16, lat32, calib, grid):
        """Apply orthographic feature transform (OFT) and sum the results"""
        signpost(header="Oft module")

        # Apply OFT and sum
        (
            ortho8,
            integral_img8,
            bbox_top_left8,
            bbox_btm_right8,
            bbox_top_right8,
            bbox_btm_left8,
        ) = (
            self.oft8.forward(device, lat8, calib, grid)
            if self.oft8 is not None
            else (None, None, None, None, None, None)
        )
        # ttnn.deallocate(lat8)
        lat8 = ttnn.to_memory_config(lat8, ttnn.DRAM_MEMORY_CONFIG) if lat8 is not None else None
        (
            ortho16,
            integral_img16,
            bbox_top_left16,
            bbox_btm_right16,
            bbox_top_right16,
            bbox_btm_left16,
        ) = (
            self.oft16.forward(device, lat16, calib, grid)
            if self.oft16 is not None
            else (None, None, None, None, None, None)
        )
        # ttnn.deallocate(lat16)
        lat16 = ttnn.to_memory_config(lat16, ttnn.DRAM_MEMORY_CONFIG) if lat16 is not None else None
        (
            ortho32,
            integral_img32,
            bbox_top_left32,
            bbox_btm_right32,
            bbox_top_right32,
            bbox_btm_left32,
        ) = (
            self.oft32.forward(device, lat32, calib, grid)
            if self.oft32 is not None
            else (None, None, None, None, None, None)
        )
        # ttnn.deallocate(lat32)
        lat32 = ttnn.to_memory_config(lat32, ttnn.DRAM_MEMORY_CONFIG) if lat32 is not None else None

        # Apply the same ortho combination logic as the reference implementation
        if (self.oft8 is not None) and (self.oft16 is not None) and (self.oft32 is not None):
            ortho = ortho8 + ortho16 + ortho32
        elif self.oft8 is not None:
            ortho = ortho8
        elif self.oft16 is not None:
            ortho = ortho16
        elif self.oft32 is not None:
            ortho = ortho32
        else:
            raise ValueError("At least one OFT module must be enabled")

        signpost(header="Oft module finished")

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
            ortho,
        )

    def forward_topdown_network(self, device, ortho):
        """Apply topdown network"""
        signpost(header="Topdown started")
        td = ortho
        for layer in self.topdown:
            logger.debug(f"Topdown layer {layer=}")
            td = layer.forward(device, td, move_to_dram=True)
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

    def preprocess(self, input_tensor):
        return self.frontend.conv1(input_tensor)

    def forward(self, device, input_tensor, calib, grid):
        signpost(header="OftNet module started")

        # Normalize input tensor
        # Commenting out as normalization is fused outside the module
        # normalized_input = self.forward_normalization(device, input_tensor)
        normalized_input = input_tensor
        # Run frontend network
        feats8, feats16, feats32 = self.frontend.forward(device, normalized_input)

        # ttnn.device.ReadDeviceProfiler(device)  # disabled because of tracy; add switch to control it later

        # Apply lateral layers
        lat8, lat16, lat32 = self.forward_lateral_layers(device, feats8, feats16, feats32)

        # ttnn.device.ReadDeviceProfiler(device)  # disabled because of tracy; add switch to control it later

        # Apply OFT transformation
        import torch  # HACK

        calib_torch = ttnn.to_torch(calib, dtype=torch.float32) if self.return_intermediates else None
        grid_torch = ttnn.to_torch(grid, dtype=torch.float32) if self.return_intermediates else None

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
            ortho,
        ) = self.forward_oft(device, lat8, lat16, lat32, calib, grid)
        # ttnn.device.ReadDeviceProfiler(device)  # disabled because of tracy; add switch to control it later

        # Apply topdown network
        td = self.forward_topdown_network(device, ortho)

        # Predict encoded outputs
        tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = self.forward_predict_encoded_outputs(device, td)
        signpost(header="OftNet finished")

        return (
            [
                (
                    feats8 if self.return_intermediates else None,
                    feats16 if self.return_intermediates else None,
                    feats32 if self.return_intermediates else None,
                    lat8 if self.return_intermediates else None,
                    lat16 if self.return_intermediates else None,
                    lat32 if self.return_intermediates else None,
                    integral_img8 if self.return_intermediates else None,
                    integral_img16 if self.return_intermediates else None,
                    integral_img32 if self.return_intermediates else None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    ortho8 if self.return_intermediates else None,
                    ortho16 if self.return_intermediates else None,
                    ortho32 if self.return_intermediates else None,
                    ortho if self.return_intermediates else None,
                    calib_torch,
                    grid_torch,
                    td if self.return_intermediates else None,
                ),
                (
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
