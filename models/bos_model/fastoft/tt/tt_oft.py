# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn.functional as F
from loguru import logger
import math

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

from ..reference.oft import EPSILON
from ..reference.utils import perspective

from .common import (
    Linear,
)

GRID_SAMPLE_NHW = 159 * 159
PAD_AMOUNT = 63
PAD_VALUE = 0.0
NUM_SLICES = 18

PRECOMPUTED_GRID_ELEMENTS_PER_POINT = 6  # For bilinear mode
PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST = 2  # For nearest mode
STANDARD_GRID_ELEMENTS_PER_POINT = 2


def split_tensor(tensor, num_slices, dim=1):
    # tensor shape: [1, total_height, 1, channels] (default split on dim=1)
    total_size = tensor.shape[dim]
    slice_size = total_size // num_slices

    splits = [slice_size] * (num_slices - 1)
    splits.append(total_size - slice_size * (num_slices - 1))
    return torch.split(tensor, splits, dim=dim)


def _prepare_grid_tensor_host(
    torch_grid,
    use_precomputed_grid,
    grid_dtype,
    input_shape_nhwc,
    grid_batching_factor,
    mode="bilinear",
    align_corners=False,
):
    """
    Common grid preparation logic for both interleaved and sharded grids.

    Args:
        torch_grid: PyTorch grid tensor
        use_precomputed_grid: Whether to use precomputed grid
        grid_dtype: Grid data type (ttnn.bfloat16 or ttnn.float32)
        input_shape_nhwc: Input shape in NHWC format (required for precomputed grid)
        grid_batching_factor: Optional batching factor for reshaping grid
        mode: Interpolation mode ("nearest" or "bilinear")

    Returns:
        ttnn tensor: Prepared grid tensor on host (not yet on device)
    """
    batch_size, grid_h, grid_w, grid_coords = torch_grid.shape
    if use_precomputed_grid:
        if input_shape_nhwc is None:
            raise ValueError("input_shape_nhwc is required for precomputed grid")

        # Create precomputed grid
        ttnn_grid_host = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
        ttnn_grid_precomputed = ttnn.prepare_grid_sample_grid(
            ttnn_grid_host,
            input_shape_nhwc,
            mode=mode,
            align_corners=align_corners,
            padding_mode="zeros",
            output_dtype=ttnn.bfloat16,
        )

        if grid_batching_factor is not None:
            # Reshape for grid batching: (N, H, W*K, elements) -> (N, H, W, elements*K)
            new_grid_w = grid_w // grid_batching_factor
            elements_per_point = (
                PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST
                if mode == "nearest"
                else PRECOMPUTED_GRID_ELEMENTS_PER_POINT
            )
            final_last_dim = elements_per_point * grid_batching_factor
            return ttnn.reshape(ttnn_grid_precomputed, (batch_size, grid_h, new_grid_w, final_last_dim))
        else:
            return ttnn_grid_precomputed
    else:
        # Create regular grid
        ttnn_grid_host = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=grid_dtype)

        if grid_batching_factor is not None:
            # Reshape for grid batching: (N, H, W*K, 2) -> (N, H, W, 2*K)
            new_grid_w = grid_w // grid_batching_factor
            new_last_dim = STANDARD_GRID_ELEMENTS_PER_POINT * grid_batching_factor
            return ttnn.reshape(ttnn_grid_host, (batch_size, grid_h, new_grid_w, new_last_dim))
        else:
            return ttnn_grid_host


def prepare_ttnn_grid(
    torch_grid,
    device,
    use_precomputed_grid,
    grid_dtype,
    input_shape_nhwc=None,
    grid_batching_factor=None,
    mode="bilinear",
    align_corners=False,
):
    """
    Prepare TTNN grid tensor from PyTorch grid.

    Args:
        torch_grid: PyTorch grid tensor
        device: TTNN device
        use_precomputed_grid: Whether to use precomputed grid
        grid_dtype: Grid data type (ttnn.bfloat16 or ttnn.float32)
        input_shape_nhwc: Input shape in NHWC format (required for precomputed grid)
        grid_batching_factor: Optional batching factor for reshaping grid
        mode: Interpolation mode ("nearest" or "bilinear")

    Returns:
        ttnn tensor: Prepared grid tensor on device
    """
    ttnn_grid_reshaped = _prepare_grid_tensor_host(
        torch_grid, use_precomputed_grid, grid_dtype, input_shape_nhwc, grid_batching_factor, mode, align_corners
    )
    return ttnn.to_device(ttnn_grid_reshaped, device)


def calculate_initialization_parameters(
    device,
    channels,
    cell_size,
    grid_height,
    feature_shape_hw,
    calib,
    grid,
    scale,
    use_precomputed_grid,
    num_slices=NUM_SLICES,
    mode="nearest",
    align_corners=True,
):
    y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
    # Expand the grid in the y dimension
    corners = grid.unsqueeze(1) + y_corners.view(-1, 1, 1, 3)

    # Project grid corners to image plane and normalize to [-1, 1]
    img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners, dtype=torch.float32)
    feature_height, feature_width = feature_shape_hw
    # Normalize to [-1, 1]
    img_size = corners.new([feature_width, feature_height]) / scale
    norm_corners = 2 * img_corners / img_size - 1

    # Get top-left and bottom-right coordinates of voxel bounding boxes
    bbox_corners = torch.cat(
        [
            torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
            torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
        ],
        dim=-1,
    )
    batch, height, depth, width, _ = bbox_corners.size()
    bbox_corners = bbox_corners.flatten(2, 3).permute(0, 2, 1, 3)
    area = (
        (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * feature_height * feature_width * 0.25
        + torch.tensor(EPSILON, dtype=bbox_corners.dtype)
    ).unsqueeze(1)

    visible = area > torch.tensor(EPSILON, dtype=area.dtype)
    # set visible to False for all bboxes that are out of bounds
    area = 1 / area
    area_nhwc = (
        torch.broadcast_to(area.permute(0, 2, 1, 3), (1, depth * width, channels, height))
        .reshape(1, depth * width, 1, channels * height)
        .permute(0, 2, 1, 3)
    )
    visible_nhwc = (
        torch.broadcast_to(visible.permute(0, 2, 1, 3), (1, depth * width, channels, height))
        .reshape(1, depth * width, 1, channels * height)
        .permute(0, 2, 1, 3)
    )

    top_left_bc = bbox_corners[..., [0, 1]]
    top_left_bc = torch.nn.functional.pad(top_left_bc, ((0, 0, 0, 0, 0, PAD_AMOUNT, 0, 0)), value=PAD_VALUE)

    btm_right_bc = bbox_corners[..., [2, 3]]
    btm_right_bc = torch.nn.functional.pad(btm_right_bc, ((0, 0, 0, 0, 0, PAD_AMOUNT, 0, 0)), value=PAD_VALUE)

    top_right_bc = bbox_corners[..., [2, 1]]
    top_right_bc = torch.nn.functional.pad(top_right_bc, ((0, 0, 0, 0, 0, PAD_AMOUNT, 0, 0)), value=PAD_VALUE)

    btm_left_bc = bbox_corners[..., [0, 3]]
    btm_left_bc = torch.nn.functional.pad(btm_left_bc, ((0, 0, 0, 0, 0, PAD_AMOUNT, 0, 0)), value=PAD_VALUE)

    batch_size, grid_h, grid_w, _ = top_left_bc.shape
    input_shape_nhwc = [batch_size, feature_height, feature_width, channels]

    top_left_bc_tt = prepare_ttnn_grid(
        top_left_bc,
        device,
        use_precomputed_grid,
        ttnn.float32,
        [batch, feature_height, feature_width, channels],
        height,
        mode=mode,
        align_corners=align_corners,
    )
    top_left_bc_tt = ttnn.reshape(top_left_bc_tt, [batch_size, 1, top_left_bc_tt.shape[1], top_left_bc_tt.shape[3]])

    btm_right_bc_tt = prepare_ttnn_grid(
        btm_right_bc,
        device,
        use_precomputed_grid,
        ttnn.float32,
        [batch, feature_height, feature_width, channels],
        height,
        mode=mode,
        align_corners=align_corners,
    )
    btm_right_bc_tt = ttnn.reshape(btm_right_bc_tt, [batch_size, 1, btm_right_bc_tt.shape[1], btm_right_bc_tt.shape[3]])

    top_right_bc_tt = prepare_ttnn_grid(
        top_right_bc,
        device,
        use_precomputed_grid,
        ttnn.float32,
        [batch, feature_height, feature_width, channels],
        height,
        mode=mode,
        align_corners=align_corners,
    )
    top_right_bc_tt = ttnn.reshape(top_right_bc_tt, [batch_size, 1, top_right_bc_tt.shape[1], top_right_bc_tt.shape[3]])

    btm_left_bc_tt = prepare_ttnn_grid(
        btm_left_bc,
        device,
        use_precomputed_grid,
        ttnn.float32,
        [batch, feature_height, feature_width, channels],
        height,
        mode=mode,
        align_corners=align_corners,
    )
    btm_left_bc_tt = ttnn.reshape(btm_left_bc_tt, [batch_size, 1, btm_left_bc_tt.shape[1], btm_left_bc_tt.shape[3]])

    top_left_bc_tt = ttnn.split(top_left_bc_tt, top_left_bc_tt.shape[2] // num_slices, dim=2)
    btm_right_bc_tt = ttnn.split(btm_right_bc_tt, btm_right_bc_tt.shape[2] // num_slices, dim=2)
    top_right_bc_tt = ttnn.split(top_right_bc_tt, top_right_bc_tt.shape[2] // num_slices, dim=2)
    btm_left_bc_tt = ttnn.split(btm_left_bc_tt, btm_left_bc_tt.shape[2] // num_slices, dim=2)

    visible_tt = ttnn.from_torch(visible_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    area = torch.nn.functional.pad(area_nhwc * visible_nhwc, ((0, 0, 0, PAD_AMOUNT, 0, 0, 0, 0)), value=PAD_VALUE)
    area = torch.split(area, area.shape[2] // num_slices, dim=2)
    area_tt = [ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device) for a in area]

    return (
        [top_left_bc_tt, btm_right_bc_tt, top_right_bc_tt, btm_left_bc_tt],
        visible_tt,
        area_tt,
        [batch, height, depth, width],
    )


class OFT:
    def __init__(
        self,
        device,
        parameters,
        input_channels,
        output_channels,
        cell_size,
        grid_height,
        features_shape_hw,
        calib,
        grid,
        scale,
        use_precomputed_grid,
        num_slices=NUM_SLICES,
        mode="nearest",
        align_corners=True,
        return_intermediates=False,
    ):
        # params for conv3d
        self.linear_weight = parameters.conv3d.weight
        self.linear_bias = parameters.conv3d.bias
        self.scale = scale
        self.use_precomputed_grid = use_precomputed_grid
        self.features_shape_hw = features_shape_hw
        self.device = device

        self.input_dtype = ttnn.bfloat16

        self.num_slices = num_slices
        self.mode = mode
        self.align_corners = align_corners

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.bbox_corners, self.visible, self.area, self.shape = calculate_initialization_parameters(
            device,
            input_channels,
            cell_size,
            grid_height,
            features_shape_hw,
            calib,
            grid,
            self.scale,
            use_precomputed_grid,
            num_slices=num_slices,
            mode=mode,
            align_corners=align_corners,
        )

        self.in_channels = self.linear_weight.shape[0]
        self.linear_weight = ttnn.reshape(
            self.linear_weight, [self.in_channels // self.shape[1], self.shape[1], output_channels]
        )
        self.linear_weight = ttnn.permute(self.linear_weight, (1, 0, 2))
        self.linear_weight = ttnn.reshape(self.linear_weight, [self.in_channels, output_channels])

        # integral_image_quantization_strategy
        # None - no quantization
        # "to_uint32" - quantize to uint32 before integral image, dequantize after
        # "to_float32" - quantize to float32 before integral image, dequantize after
        self.integral_image_quantization_strategy = None
        logger.info(f"Integral image quantization strategy: {self.integral_image_quantization_strategy}")
        if self.integral_image_quantization_strategy == None:
            self.prescaler = ttnn.from_torch(torch.tensor(1024 * 1024), device=device, dtype=ttnn.bfloat16)
            self.postscaler = ttnn.from_torch(torch.tensor(1 / 1024 / 1024), device=device, dtype=ttnn.bfloat16)

        # Initialize sharding and linear layer configurations
        linear_pt = {
            "in_channels": self.in_channels,
            "out_channels": output_channels,
            "nhw": (GRID_SAMPLE_NHW + PAD_AMOUNT) // self.num_slices,
            "height_sharding": True,
        }

        self._setup_sharding_configs()
        self.linear_layer = Linear(self.linear_weight, self.linear_bias, linear_pt)
        out_initial = torch.zeros([1, 1, GRID_SAMPLE_NHW + PAD_AMOUNT, self.output_channels], dtype=torch.float32)
        self.ortho_feats = ttnn.from_torch(
            out_initial, self.input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.return_intermediates = return_intermediates

    def _setup_sharding_configs(self):
        """Setup sharding configurations for slicing operations"""
        compute_grid = self.device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
        self.sharding_strategy = "height"

        # Sharding parameters for slicing
        self.slice_memory_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            if self.sharding_strategy == "height"
            else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

    def forward(self, device, features, calib, grid):
        if use_signpost:
            signpost(header="OFT block started")

        features = ttnn.reshape(features, [1, self.features_shape_hw[0], self.features_shape_hw[1], -1])
        if features.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
            features = ttnn.to_layout(features, ttnn.TILE_LAYOUT)

        if self.integral_image_quantization_strategy is None:
            integral_image = ttnn_integral_image_channel_last(features)
        elif self.integral_image_quantization_strategy == "to_uint32":
            features = ttnn.mul(features, self.prescaler, dtype=ttnn.bfloat16)
            features = ttnn.typecast(features, ttnn.uint32)
            integral_image = ttnn_integral_image_channel_last(features)
            integral_image = ttnn.typecast(integral_image, ttnn.bfloat16)
            integral_image = ttnn.mul(integral_image, self.postscaler, dtype=ttnn.bfloat16)
        elif self.integral_image_quantization_strategy == "to_float32":
            features = ttnn.typecast(features, ttnn.float32)
            integral_image = ttnn_integral_image_channel_last(features)
            integral_image = ttnn.typecast(integral_image, ttnn.bfloat16)

        if integral_image.get_layout() == ttnn.TILE_LAYOUT:
            integral_image = ttnn.to_layout(integral_image, ttnn.ROW_MAJOR_LAYOUT)

        integral_image = ttnn.to_memory_config(integral_image, ttnn.L1_MEMORY_CONFIG)

        grid_size = self.device.compute_with_storage_grid_size()
        core_grid = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)

        n, h, w, in_ch = [1, 1, GRID_SAMPLE_NHW + PAD_AMOUNT, self.in_channels]  # features.shape

        out_ch = self.output_channels

        # Calculate dynamic configurations based on tensor dimensions
        grid_sample_shard_height = (
            math.ceil(n * h * w // (self.num_slices * ttnn.TILE_SIZE) / (core_grid.y * core_grid.x)) * ttnn.TILE_SIZE
        )
        grid_sample_shard_width = math.ceil(in_ch // ttnn.TILE_SIZE) * ttnn.TILE_SIZE

        logger.debug(
            f"Grid sample shard dimensions - height: {grid_sample_shard_height}, width: {grid_sample_shard_width}, "
            f"core_grid: {core_grid}"
        )

        grid_sample_memory_config = ttnn.create_sharded_memory_config(
            (grid_sample_shard_height, grid_sample_shard_width),
            core_grid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )


        grid_memory_config = ttnn.create_sharded_memory_config(
            (grid_sample_shard_height, 32),
            core_grid,
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        for i in range(self.num_slices):
            # Top left corner slice
            top_left_grid = ttnn.to_memory_config(self.bbox_corners[0][i], grid_memory_config)
            vox_feats_slice = ttnn.grid_sample(
                integral_image,
                top_left_grid,
                use_precomputed_grid=self.use_precomputed_grid,
                batch_output_channels=True,
                mode=self.mode,
                align_corners=self.align_corners,
                memory_config=grid_sample_memory_config,
            )
            ttnn.deallocate(top_left_grid)
            vox_feats_slice = ttnn.to_layout(vox_feats_slice, ttnn.TILE_LAYOUT)

            # Top right corner slice
            top_right_grid = ttnn.to_memory_config(self.bbox_corners[2][i], grid_memory_config)
            top_right_slice = ttnn.grid_sample(
                integral_image,
                top_right_grid,
                use_precomputed_grid=self.use_precomputed_grid,
                batch_output_channels=True,
                mode=self.mode,
                align_corners=self.align_corners,
                memory_config=grid_sample_memory_config,
            )
            ttnn.deallocate(top_right_grid)
            top_right_slice = ttnn.to_layout(top_right_slice, ttnn.TILE_LAYOUT)
            ttnn.sub_(vox_feats_slice, top_right_slice)
            ttnn.deallocate(top_right_slice)

            # Bottom right corner slice
            btm_right_grid = ttnn.to_memory_config(self.bbox_corners[1][i], grid_memory_config)
            btm_right_slice = ttnn.grid_sample(
                integral_image,
                btm_right_grid,
                use_precomputed_grid=self.use_precomputed_grid,
                batch_output_channels=True,
                mode=self.mode,
                align_corners=self.align_corners,
                memory_config=grid_sample_memory_config,
            )
            ttnn.deallocate(btm_right_grid)
            btm_right_slice = ttnn.to_layout(btm_right_slice, ttnn.TILE_LAYOUT)
            ttnn.add_(vox_feats_slice, btm_right_slice)
            ttnn.deallocate(btm_right_slice)

            # Bottom left corner slice
            btm_left_grid = ttnn.to_memory_config(self.bbox_corners[3][i], grid_memory_config)
            btm_left_slice = ttnn.grid_sample(
                integral_image,
                btm_left_grid,
                use_precomputed_grid=self.use_precomputed_grid,
                batch_output_channels=True,
                mode=self.mode,
                align_corners=self.align_corners,
                memory_config=grid_sample_memory_config,
            )
            ttnn.deallocate(btm_left_grid)
            btm_left_slice = ttnn.to_layout(btm_left_slice, ttnn.TILE_LAYOUT)
            ttnn.sub_(vox_feats_slice, btm_left_slice)
            ttnn.deallocate(btm_left_slice)

            area_slice = self.area[i]
            area_slice = ttnn.to_memory_config(area_slice, grid_sample_memory_config)
            ttnn.mul_(vox_feats_slice, area_slice)
            vox_feats_slice = ttnn.move(vox_feats_slice)
            vox_feats_slice = self.linear_layer(vox_feats_slice, device)

            ttnn.sharded_to_interleaved_partial(
                vox_feats_slice, self.ortho_feats, self.num_slices, i, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(vox_feats_slice)

        ortho_feats = self.ortho_feats[:, :, : w - PAD_AMOUNT, :]
        # this is used as intermediate tensor for tracking pcc over model
        # if removing intermediate tensors, remove this but call ttnn.deallocate(integral_image)
        integral_image = ttnn.to_torch(integral_image).permute(0, 3, 1, 2) if self.return_intermediates else None

        if use_signpost:
            signpost(header="OFT block ended")
        # return ortho_feats
        return (
            ortho_feats,
            integral_image,
            self.bbox_corners[0],
            self.bbox_corners[1],
            self.bbox_corners[2],
            self.bbox_corners[3],
        )


def ttnn_integral_image_channel_last(features_nhwc):
    assert len(features_nhwc.shape) == 4, "Input tensor must be 4D"
    assert features_nhwc.shape[0] == 1, "Batch size must be 1"
    return ttnn.experimental.intimg(features_nhwc)
