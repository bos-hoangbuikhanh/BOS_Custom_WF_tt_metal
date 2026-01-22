# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from loguru import logger

from collections import namedtuple

from .common import Conv

ObjectData = namedtuple("ObjectData", ["classname", "position", "dimensions", "angle", "score"])


def gaussian_kernel(sigma=1.0, trunc=2.0):
    width = round(trunc * sigma)
    x = torch.arange(-width, width + 1).float() / sigma
    kernel1d = torch.exp(-0.5 * x**2)
    kernel2d = kernel1d.view(1, -1) * kernel1d.view(-1, 1)

    return kernel2d / kernel2d.sum()


class TTObjectEncoder:
    def __init__(
        self,
        device,
        parameters,
        grid,
        classnames=["Car"],
        pos_std=[0.5, 0.36, 0.5],
        log_dim_mean=[[0.42, 0.48, 1.35]],
        log_dim_std=[[0.085, 0.067, 0.115]],
        sigma=1.0,
        nms_thresh=0.05,
        return_intermediates=False,
    ):
        self.classnames = classnames
        self.nclass = len(classnames)
        self.pos_std = torch.tensor(pos_std)
        self.log_dim_mean = torch.tensor(log_dim_mean)
        self.log_dim_std = torch.tensor(log_dim_std)

        # Compute the center of each grid cell once
        # grid has shape [H, W, 3] representing (x, y, z) coordinates in channel-last format
        centers = grid[1:, 1:, :] + grid[:-1, :-1, :]
        centers = centers / 2.0
        # Permute to NCHW: [H-1, W-1, 3] -> [3, H-1, W-1]
        centers = centers.permute(2, 0, 1)

        # Reshape for NCHW broadcasting: [3] -> [1, 3, 1, 1] and [1, 3] -> [1, 3, 1, 1]
        pos_std_reshaped = self.pos_std.view(1, 3, 1, 1)
        log_dim_std_reshaped = self.log_dim_std.view(1, 3, 1, 1)
        log_dim_mean_reshaped = self.log_dim_mean.view(1, 3, 1, 1)

        self.centers = ttnn.from_torch(
            centers,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
        )
        self.pos_std_reshaped = ttnn.from_torch(
            pos_std_reshaped,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
        )
        self.log_dim_mean_reshaped = ttnn.from_torch(
            log_dim_mean_reshaped,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
        )
        self.log_dim_std_reshaped = ttnn.from_torch(
            log_dim_std_reshaped,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
        )

        self.sigma = sigma
        self.nms_thresh = nms_thresh  # is there a typo in reference code? nms_thresh is passed but heatmaps is called with default value 0.05
        self.nms_conv = Conv(
            parameters.nms_conv,
            parameters.layer_args,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            act_block_h=8*32,
        )
        self.max_peaks = 50
        self.return_intermediates = return_intermediates

    def decode(self, device, heatmaps, pos_offsets, dim_offsets, ang_offsets):
        tt_positions = self._decode_positions(device, pos_offsets)
        tt_dimensions = self._decode_dimensions(device, dim_offsets)
        tt_angles = self._decode_angles(device, ang_offsets)
        tt_indices, tt_scores, torch_smoothed, torch_mp = self._decode_heatmaps(device, heatmaps)

        return (
            [tt_indices, tt_scores, tt_positions, tt_dimensions, tt_angles],
            [torch_smoothed, torch_mp],
            ("indices", "scores", "positions", "dimensions", "angles"),
            ("smoothed", "mp"),
        )

    def decoder_postprocess(self, tt_indices, tt_scores, tt_positions, tt_dimensions, tt_angles):
        # TODO(mbezulj): figure out ttnn way to handle this in a performant manner

        # Convert all ttnn tensors to torch
        n = 1
        c, h, w = tt_scores.shape
        indices_torch = ttnn.to_torch(tt_indices, dtype=torch.int64).permute(0, 3, 1, 2).view(n, c, h, w)
        scores_torch = ttnn.to_torch(tt_scores, dtype=torch.float32)
        # Permute positions and dimensions from NCHW to HWC for indexing
        positions_torch = ttnn.to_torch(tt_positions, dtype=torch.float32).permute(0, 2, 3, 1)
        dimensions_torch = ttnn.to_torch(tt_dimensions, dtype=torch.float32).permute(0, 2, 3, 1)
        angles_torch = ttnn.to_torch(tt_angles, dtype=torch.float32)

        # Find the pixels which correspond to the maximum indices
        _, height, width = scores_torch.size()
        flat_inds = torch.arange(height * width).type_as(indices_torch).view(height, width)
        peaks_torch = (flat_inds == indices_torch) & (scores_torch > self.nms_thresh)

        # Keep only the top N peaks
        if peaks_torch.long().sum() > self.max_peaks:
            scores = scores_torch[peaks_torch.squeeze(0)]
            scores, _ = torch.sort(scores, descending=True)
            peaks_torch = peaks_torch & (scores_torch > scores[self.max_peaks - 1])
        peaks_torch = peaks_torch[0]

        logger.debug(f"tt_peaks {peaks_torch.long().sum()}")

        classids_torch = torch.nonzero(peaks_torch)[:, 0]
        scores_torch = scores_torch[peaks_torch]
        positions_torch = positions_torch[peaks_torch]
        dimensions_torch = dimensions_torch[peaks_torch]
        angles_torch = angles_torch[peaks_torch]

        return scores_torch, classids_torch, positions_torch, dimensions_torch, angles_torch

    def _decode_heatmaps(self, device, heatmaps):
        max_inds, smoothed, mp = self._non_maximum_suppression(device, heatmaps)
        scores = heatmaps
        # classids = torch.nonzero(peaks)[:, 0] #moved to level above
        return max_inds, scores, smoothed, mp

    def _decode_positions(self, device, pos_offsets):
        # Un-normalize grid offsets using precomputed centers
        positions = pos_offsets * self.pos_std_reshaped + self.centers
        return positions

    def _decode_dimensions(self, device, dim_offsets):
        # Keep NCHW layout
        coef = dim_offsets * self.log_dim_std_reshaped + self.log_dim_mean_reshaped
        dimensions = ttnn.exp(coef)
        return dimensions

    def _decode_angles(self, device, angle_offsets):
        cos = angle_offsets[:, 0, :, :]
        sin = angle_offsets[:, 1, :, :]
        atan2 = ttnn.atan2(sin, cos)
        return atan2

    def _non_maximum_suppression(self, device, heatmaps):
        heatmaps_4d = ttnn.unsqueeze(heatmaps, 0)
        n, c, h, w = heatmaps_4d.shape
        heatmaps_4d = ttnn.permute(heatmaps_4d, (0, 2, 3, 1))  # NHWC for conv/maxpool

        smoothed, out_h, out_w = self.nms_conv(device, heatmaps_4d)
        torch_smoothed = ttnn.to_torch(smoothed, dtype=torch.float32) if self.return_intermediates else None


        mp, indices = ttnn.max_pool2d(
            input_tensor=smoothed,
            batch_size=n,
            input_h=h,
            input_w=w,
            channels=c,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED if not smoothed.is_sharded() else None,
            ceil_mode=False,
            deallocate_input=False,
            reallocate_halo_output=True,
            return_indices=True,
        )
        # we need to convert indices to L1 and TILE layout to be able to copy it to host allocated tensor
        # namely without this conversion, indices tensor has a padded shape != logical shape
        # and allocate host tensor api cannot take padded shape in addition to logical shape
        indices = ttnn.to_memory_config(indices, ttnn.L1_MEMORY_CONFIG)
        indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)
        torch_mp = ttnn.to_torch(mp, dtype=torch.float32).permute(0, 3, 1, 2) if self.return_intermediates else None

        return indices, torch_smoothed, torch_mp

    def create_objects(self, scores, classids, positions, dimensions, angles):
        """Separate method to create ObjectData list from tensors"""
        objects = []
        for score, cid, pos, dim, ang in zip(scores, classids, positions, dimensions, angles):
            objects.append(ObjectData(self.classnames[cid], pos, dim, ang, score))
        return objects
