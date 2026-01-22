# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.bos_model.ufld_v2.ttnn.common import TtnnUFLDV2Conv2D
from models.bos_model.ufld_v2.ttnn.ttnn_resnet_34 import TtnnResnet34


class TtnnUFLDv2:
    def __init__(self, conv_args, conv_pth, device, model_type="tusimple"):
        self.input_height = 320
        self.input_width = 800
        self.num_grid_row = 100
        self.num_cls_row = 56
        self.num_grid_col = 100
        self.num_cls_col = 41
        self.num_lane_on_row = 4
        self.num_lane_on_col = 4
        self.use_aux = False
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        self.input_dim = self.input_height // 32 * self.input_width // 32 * 8
        self.conv_pth = conv_pth

        if model_type == "culane":
            slice_config = None  # AUTO: let ttnn choose slicing (often DRAM-slicing path on this model)
            resnet_maxpool_dram = True
        else:  # "tusimple"
            slice_config = ttnn.Conv2dL1FullSliceConfig
            resnet_maxpool_dram = False

        self.res_model = TtnnResnet34(
            conv_args,
            conv_pth.res_model,
            device=device,
            slice_config=slice_config,
            maxpool_dram=resnet_maxpool_dram,
        )
        self.pool = TtnnUFLDV2Conv2D(conv_args.pool, conv_pth.pool, activation=None, device=device)

        self.device = device

        self.has_layernorm = False
        self.layernorm_weight = None
        self.layernorm_bias = None

        try:
            ln = getattr(self.conv_pth.cls, "layernorm")
            self.has_layernorm = ln is not None
        except (AttributeError, KeyError):
            ln = None
            self.has_layernorm = False

        if self.has_layernorm:
            self.layernorm_weight = ttnn.to_layout(ln.weight, ttnn.TILE_LAYOUT)
            self.layernorm_bias = ttnn.to_layout(ln.bias, ttnn.TILE_LAYOUT)

    def __call__(self, input, batch_size=1, grid_size=(5, 4), tile_size=32, linear_memory_config="L1_INTERLEAVED"):
        fea = self.res_model(input, batch_size=batch_size)
        fea, out_h, out_w = self.pool(fea)

        if linear_memory_config == "L1_WIDTH_SHARDED":
            linear_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            shard_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                    )
                }
            )
            mem_config = ttnn.create_sharded_memory_config_(
                ttnn.Shape([fea.shape[-2] * fea.shape[-1], tile_size]),
                shard_grid,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
            )
            fea = ttnn.to_memory_config(fea, mem_config)

        elif linear_memory_config == "L1_INTERLEAVED":
            linear_memory_config = ttnn.L1_MEMORY_CONFIG
            fea = ttnn.to_memory_config(fea, ttnn.L1_MEMORY_CONFIG)

        fea = ttnn.permute(fea, (0, 1, 3, 2))
        fea = ttnn.reshape(fea, (fea.shape[0], fea.shape[1], 1, fea.shape[2] * fea.shape[3]))
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # add LayerNorm (for CULane34)
        if self.has_layernorm:
            fea = ttnn.layer_norm(
                fea,
                weight=self.layernorm_weight,
                bias=self.layernorm_bias,
                epsilon=1e-12,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=compute_config,
            )

        out = ttnn.linear(
            fea,
            self.conv_pth.cls.linear_1.weight,
            bias=self.conv_pth.cls.linear_1.bias,
            memory_config=linear_memory_config,
            compute_kernel_config=compute_config,
        )
        out = ttnn.relu(out)
        out = ttnn.linear(
            out,
            self.conv_pth.cls.linear_2.weight,
            bias=self.conv_pth.cls.linear_2.bias,
            memory_config=linear_memory_config,
            compute_kernel_config=compute_config,
        )
        return out
