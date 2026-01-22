# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..reference import resnet
from ..reference.resnet import FrontendMode
from ..reference.oft import OFT

from loguru import logger
from enum import Enum


class OftMode(Enum):
    ORIGINAL = 0
    OFT8 = 1
    OFT16 = 2
    OFT32 = 3

    def __str__(self):
        return self.name.lower()


def get_num_topdown_layers(oft_mode: OftMode):
    if oft_mode == OftMode.ORIGINAL:
        return 8
    else:
        return 2


class OftNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=6.0,
        dtype=torch.float32,
        oft_mode=OftMode.ORIGINAL,
        frontend_mode=FrontendMode.ORIGINAL,
    ):
        super().__init__()

        # Construct frontend network
        assert frontend in ["resnet18", "resnet34"], "unrecognised frontend"
        self.frontend = getattr(resnet, frontend)(pretrained=False, dtype=dtype, frontend_mode=frontend_mode)

        # Lateral layers convert resnet outputs to a common feature size
        self.lat8 = (
            nn.Conv2d(128, 256, 1, dtype=dtype) if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT8) else None
        )
        self.lat16 = (
            nn.Conv2d(256, 256, 1, dtype=dtype) if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT16) else None
        )
        self.lat32 = (
            nn.Conv2d(512, 256, 1, dtype=dtype) if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT32) else None
        )
        self.bn8 = nn.BatchNorm2d(256) if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT8) else None
        self.bn16 = nn.BatchNorm2d(256) if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT16) else None
        self.bn32 = nn.BatchNorm2d(256) if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT32) else None

        # Orthographic feature transforms
        input_channels = 256  # Channels from lateral layers
        output_channels = 128  # Updated from 256 to 128 as per user's change

        self.oft8 = (
            OFT(input_channels, output_channels, grid_res, grid_height, 1 / 8.0, dtype=dtype)
            if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT8)
            else None
        )
        self.oft16 = (
            OFT(input_channels, output_channels, grid_res, grid_height, 1 / 16.0, dtype=dtype)
            if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT16)
            else None
        )
        self.oft32 = (
            OFT(input_channels, output_channels, grid_res, grid_height, 1 / 32.0, dtype=dtype)
            if (oft_mode == OftMode.ORIGINAL or oft_mode == OftMode.OFT32)
            else None
        )

        # Topdown network
        self.topdown = nn.Sequential(*[resnet.BasicBlock(128, 128, dtype=dtype) for _ in range(topdown_layers)])

        # Detection head
        self.head = nn.Conv2d(128, num_classes * 9, kernel_size=3, padding=1, dtype=dtype)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=dtype))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=dtype))

        self.dtype = dtype
        self.oft_mode = oft_mode

        # Convert all parameters and buffers to the specified dtype using PyTorch's to() method
        self.to(dtype)

    def forward(self, image, calib, grid):
        if image.dtype != self.dtype:
            logger.warning(f"Input image dtype {image.dtype} does not match model dtype {self.dtype}, converting")
            image = image.to(self.dtype)
        if calib.dtype != self.dtype:
            logger.warning(f"Input calib dtype {calib.dtype} does not match model dtype {self.dtype}, converting")
            calib = calib.to(self.dtype)
        if grid.dtype != self.dtype:
            logger.warning(f"Input grid dtype {grid.dtype} does not match model dtype {self.dtype}, converting")
            grid = grid.to(self.dtype)

        # Normalize by mean and std-dev
        # Commenting out as normalization is fused outside the module
        # image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        # Run frontend network
        feats8, feats16, feats32 = self.frontend(image)

        # Apply lateral layers to convert image features to common feature size
        lat8 = F.relu(self.bn8(self.lat8(feats8))) if self.bn8 is not None and self.lat8 is not None else None
        lat16 = F.relu(self.bn16(self.lat16(feats16))) if self.bn16 is not None and self.lat16 is not None else None
        lat32 = F.relu(self.bn32(self.lat32(feats32))) if self.bn32 is not None and self.lat32 is not None else None

        # Apply OFT and sum
        ortho8, integral_img8, bbox_top_left8, bbox_btm_right8, bbox_top_right8, bbox_btm_left8 = (
            self.oft8(lat8, calib, grid) if self.oft8 is not None else (None, None, None, None, None, None)
        )
        ortho16, integral_img16, bbox_top_left16, bbox_btm_right16, bbox_top_right16, bbox_btm_left16 = (
            self.oft16(lat16, calib, grid) if self.oft16 is not None else (None, None, None, None, None, None)
        )
        ortho32, integral_img32, bbox_top_left32, bbox_btm_right32, bbox_top_right32, bbox_btm_left32 = (
            self.oft32(lat32, calib, grid) if self.oft32 is not None else (None, None, None, None, None, None)
        )

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

        # Apply topdown network
        topdown = self.topdown(ortho)

        # Predict encoded outputs
        batch, _, depth, width = topdown.size()
        outputs = self.head(topdown).view(batch, -1, 9, depth, width)
        scores, pos_offsets, dim_offsets, ang_offsets = torch.split(outputs, [1, 3, 3, 2], dim=2)

        # return scores.squeeze(2), pos_offsets, dim_offsets, ang_offsets
        return (
            [
                feats8,
                feats16,
                feats32,
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
                calib,
                grid,
                topdown,
            ],
            scores.squeeze(2),
            pos_offsets,
            dim_offsets,
            ang_offsets,
        )
