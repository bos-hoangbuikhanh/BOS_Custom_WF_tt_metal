# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PyTest for the TT-NN implementation of Encoder (TemporalSelfAttention + Norm + SpatialCrossAttention + Norm + FFN).

• Compares TT results against the reference PyTorch implementation.
• Uses deterministic random inputs generated once per test run.
• Verifies shape equality and numerical closeness (PCC > 0.98).
"""

import time
import torch
import pytest
import ttnn
from loguru import logger

from bos_metal import device_box, engine
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.bricks.registry import NORM_LAYERS

from test.SSR.reference.encoder import *
from test.SSR.reference.temporal_self_attention import *
from test.SSR.reference.spatial_cross_attention import *
from test.SSR.tt.encoder import *
from test.SSR.tt.temporal_self_attention import *
from test.SSR.tt.spatial_cross_attention import *
from test.common import *

from test.utils import pt2tt, compare_tensors

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
torch.manual_seed(0)

BATCH_SIZE     = 1
NUM_CAMS       = 6
# SRC_SEQ_LEN    = 80 * 80     # dynamic length governed by bev_mask
BEV_H, BEV_W   = 100, 100
SRC_SEQ_LEN    = BEV_H * BEV_W
MAX_LEN        = 3_680
TGT_SEQ_LEN    = 240
NUM_CHANNELS   = 256
NORM_LAYERS.register_module('LN_tt', module=op.LayerNorm, force=True)

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def device():
    """Open a Tenstorrent device for the whole test session, close at teardown."""
    device_box.open(enable_program_cache=True)           # default device_id = 0
    dev = device_box.get()
    yield dev
    device_box.close()


# @pytest.fixture
def random_inputs(device):
    """Generate deterministic PyTorch inputs and their TT equivalents."""

    # ------------------- PyTorch tensors ---------------------------------- #
    ref_bev_query = torch.randn(SRC_SEQ_LEN, BATCH_SIZE, NUM_CHANNELS)
    ref_key = torch.randn(NUM_CAMS, TGT_SEQ_LEN, BATCH_SIZE, NUM_CHANNELS)
    ref_value = ref_key
    ref_bev_pos = torch.randn(SRC_SEQ_LEN, BATCH_SIZE, NUM_CHANNELS)
    ref_prev_bev = None 
    ref_shift = torch.zeros(1, 2)
    
    bev_h, bev_w = BEV_H, BEV_W
    level_start_index       = torch.tensor([0])
    spatial_shapes          = torch.tensor([[12, 20]])
    
    img_metas_pt = [
        dict(
            lidar2img=[torch.randn(4, 4) for _ in range(NUM_CAMS)],
            img_shape=[(384, 640, 3) for _ in range(NUM_CAMS)],
        )
    ]

    img_metas_tt = [
        dict(
            lidar2img=[
                ttnn.typecast(
                    ttnn.from_torch(
                        tensor=t,
                        dtype=ttnn.float32,
                        # shape=list(can_bus.shape),
                        layout=ttnn.Layout.TILE,
                        device=device_box.get()
                    ),
                    ttnn.bfloat16
                ) for t in img_metas_pt[0]['lidar2img']
            ],
            img_shape=[(384, 640, 3) for _ in range(NUM_CAMS)],
        )
    ]

    ref_input = dict(
        bev_query            = ref_bev_query,
        key                  = ref_key,
        value                = ref_value,
        bev_pos              = ref_bev_pos,
        prev_bev             = ref_prev_bev,
        spatial_shapes       = spatial_shapes,
        level_start_index    = level_start_index,
        bev_h                = bev_h,
        bev_w                = bev_w,
        img_metas            = img_metas_pt,    # kwarg for BEVFormerEncoder
        shift                = ref_shift,
    )

    # ------------------- TT-NN tensors ------------------------------------ #
    bev_query   = pt2tt(ref_bev_query.permute(1, 0, 2),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    # memory_config= op.ShardedMemConfig(
                    #     (320, 256), (5, 4), 'height', 
                    #     as_shard_shape=True),
                    device=device)
    key    = pt2tt(ref_key.permute(2, 0, 1, 3),   # B, C, T, D
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    device=device)
    value    = pt2tt(ref_key.permute(2, 0, 1, 3),   # B, C, T, D
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    device=device)
    bev_pos = pt2tt(ref_bev_pos.permute(1, 0, 2),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    device=device)
    shift = pt2tt(ref_shift,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device)

    tt_input = dict(
        bev_query            = bev_query,
        key                  = key,
        value                = value,
        bev_pos              = bev_pos,
        prev_bev             = None,
        spatial_shapes       = spatial_shapes,
        level_start_index    = level_start_index,
        bev_h                = bev_h,
        bev_w                = bev_w,
        img_metas            = img_metas_tt,    # kwarg for BEVFormerEncoder
        shift                = shift,
    )

    return ref_input, tt_input


# --------------------------------------------------------------------------- #
# Test                                                                        #
# --------------------------------------------------------------------------- #
def test_encoder(device):
    """End-to-end comparison between PyTorch and TT-NN BEVFormerEncoder."""

    ref_input, tt_input = random_inputs(device)

    # ------------------- Build reference model ---------------------------- #
    cfg=dict(
        type='BEVFormerEncoder',
        num_layers=3,
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        num_points_in_pillar=4,
        return_intermediate=False,
        transformerlayers=dict(
            type='BEVFormerLayer',
            attn_cfgs=[
                dict(
                    type='TemporalSelfAttention',
                    embed_dims=256,
                    num_levels=1),
                dict(
                    type='SpatialCrossAttention',
                    pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                    deformable_attention=dict(
                        type='MSDeformableAttention3D',
                        embed_dims=256,
                        num_points=8,
                        num_levels=1),
                    embed_dims=256,
                    max_len=MAX_LEN,
                )
            ],
            ffn_cfgs=dict(type='FFN', embed_dims=256),
            norm_cfg=dict(type='LN'),
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                'ffn', 'norm'),
            bev_h=BEV_H,
            bev_w=BEV_W
        ),
        bev_h=BEV_H,
        bev_w=BEV_W,
        max_len=MAX_LEN,
    )
    ref_model = build_transformer_layer_sequence(cfg)
    ref_model.eval()

    # ------------------- Build TT-NN model -------------------------------- #
    cfg_ttnn = cfg.copy()
    cfg_ttnn['type'] = 'BEVFormerEncoder_tt'
    cfg_ttnn['transformerlayers']['type'] = 'BEVFormerLayer_tt'
    cfg_ttnn['transformerlayers']['attn_cfgs'][0]['type'] = 'TemporalSelfAttention_tt'
    cfg_ttnn['transformerlayers']['attn_cfgs'][1]['type'] = 'SpatialCrossAttention_tt'
    cfg_ttnn['transformerlayers']['attn_cfgs'][1]['deformable_attention']['type'] = 'MSDeformableAttention3D_tt'
    cfg_ttnn['transformerlayers']['ffn_cfgs']['type'] = 'FFN_tt'
    cfg_ttnn['transformerlayers']['norm_cfg']['type'] = 'LN_tt'
    model = build_transformer_layer_sequence(cfg_ttnn)

    # Load processed state dict
    state_dict = engine.ModelProcessor(ref_model).process_state_dict(**ref_input)
    model.load_state_dict(state_dict, strict=False)
    model.reference_points = pt2tt(ref_model.reference_points, device=device)
    model.reference_points = ttnn.squeeze(model.reference_points, -1)

    # ------------------- L1 Optimization ---------------------------------- #
    import math
    from test.common import MyDict
    program_config = MyDict({
        "self_attn": MyDict({
            "value_proj": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(5, 4),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=math.ceil(BEV_H * BEV_W / 20 / 32) * 2,
                # per_core_M=20,
                per_core_N=8,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            ), 
            "output_proj": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(5, 4),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=8,
                per_core_M=math.ceil(BEV_H * BEV_W / 20 / 32),
                # per_core_M=10,
                per_core_N=8,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            ),
        }),
        "cross_attn": MyDict({
            "output_proj":ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(5, 4),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=8,
                per_core_M=math.ceil(BEV_H * BEV_W / 20 / 32),
                # per_core_M=10,
                per_core_N=8,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            )
        }),
        "norm": MyDict({}),
        "ffn": MyDict({
            0: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(5, 4),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=8,
                per_core_M=math.ceil(BEV_H * BEV_W / 20 / 32),
                # per_core_M=10,
                per_core_N=16,
                fuse_batch=True,
                fused_activation=ttnn.UnaryOpType.RELU,
                mcast_in0=False,
            ),
            1: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(5, 4),
                in0_block_w=16,
                out_subblock_h=1,
                out_subblock_w=8,
                per_core_M=math.ceil(BEV_H * BEV_W / 20 / 32),
                # per_core_M=10,
                per_core_N=8,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            ),
        })
    })
    memory_config = MyDict({
        "bev_query": op.ShardedMemConfig(
            # shape=(512, 256),
            shape=(math.ceil(BEV_H * BEV_W / 20 / 32) * 32, 256),
            # shape=(320, 256),
            core_grid=(5, 4),
            strategy='height',
            as_shard_shape=True
        ),
        "self_attn": MyDict({
            "value": op.ShardedMemConfig(
                # (640, 256), 
                (math.ceil(BEV_H * BEV_W / 20 / 32) * 32 * 2, 256),
                (5, 4), 
                "height", 
                as_shard_shape=True
            ),
            "value_proj": ttnn.DRAM_MEMORY_CONFIG,
            "output_proj": ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        }),
        "cross_attn": MyDict({
            "output_proj": ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        }),
        "ffn": ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    })

    # ------------------- Forward passes ----------------------------------- #
    tt_out  = model(**tt_input, memory_config=memory_config, program_config=program_config)
    ref_out = ref_model(**ref_input)
    ttnn.synchronize_device(device)
    compare_tensors(ref_out, tt_out)

    # ------------------- Assertions --------------------------------------- #
    ref_out = ref_model(**ref_input)
    assert ref_out.shape == tt_out.shape, "Output shape mismatch"
    passed, _ = compare_tensors(ref_out, tt_out, pcc_thresh=0.98)
    assert passed, f"PCC below threshold"
    # ttnn.deallocate(tt_out)
    breakpoint()
    
    num_iter = 10
    for i in range(num_iter):
        ref_input, tt_input = random_inputs(device)
        ref_input['prev_bev'] = ref_out.permute(1, 0, 2)
        tt_input['prev_bev'] = ttnn.to_memory_config(tt_out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tt_out)
        ref_out = ref_model(**ref_input)
        
        print('#', '-' * 50)
        st = time.time()
        tt_out = model(**tt_input, memory_config=memory_config, program_config=program_config)
        ttnn.synchronize_device(device)
        en = time.time()
        compare_tensors(ref_out, tt_out)
        print('#', '-' * 50)
        # ttnn.deallocate(tt_out)
        avg_exec_time = (en - st) * 1000 # in ms
        print(f"Iter {i+1}/{num_iter} done, runtime: {avg_exec_time:.4f} ms")

    print(f"Average time: {avg_exec_time:.2f} ms | {1000/(avg_exec_time):.2f} FPS")
