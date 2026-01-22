import ttnn
from test.common import MyDict
from bos_metal import op
core_grid = ttnn.CoreRangeSet([
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))
])

memory_config = MyDict({
    'se_layer': {
        "x": op.ShardedMemConfig((64, 2016), (5, 4), 'block', orientation=ttnn.ShardOrientation.COL_MAJOR, as_shard_shape=True),
        "mlp_reduce": ttnn.L1_MEMORY_CONFIG,
        "mlp_expand": ttnn.L1_MEMORY_CONFIG,
    },
    'concat': {
        'bev_x': ttnn.L1_MEMORY_CONFIG,
        'out_se': ttnn.L1_MEMORY_CONFIG,
        'out': ttnn.DRAM_MEMORY_CONFIG,
    },
    'tokenlearner': {
        "mlp": {
            "input": op.ShardedMemConfig((512, 512), (5, 4), 'height', as_shard_shape=True),
            "fc1": op.ShardedMemConfig((512, 64), (5, 4), 'height', as_shard_shape=True),
            "fc2": op.ShardedMemConfig((512, 32), (5, 4), 'height', as_shard_shape=True),
        },
    },
    'latent_decoder': {
        "self_attn": {
            "q_input": ttnn.L1_MEMORY_CONFIG,
            "kv_input": ttnn.L1_MEMORY_CONFIG,
            "in_proj_q": ttnn.L1_MEMORY_CONFIG,
            "in_proj_kv": ttnn.L1_MEMORY_CONFIG,
            "qkv_permute": ttnn.L1_MEMORY_CONFIG,
            "qkv_split": ttnn.L1_MEMORY_CONFIG,
        },
        "norm": ttnn.L1_MEMORY_CONFIG,
        "ffn": ttnn.L1_MEMORY_CONFIG,
    },
    'way_decoder': {
        "cross_attn": {
            "q_input": ttnn.L1_MEMORY_CONFIG,
            "kv_input": ttnn.L1_MEMORY_CONFIG,
            "in_proj_q": ttnn.L1_MEMORY_CONFIG,
            "in_proj_kv": ttnn.L1_MEMORY_CONFIG,
            "qkv_split": ttnn.L1_MEMORY_CONFIG,
            "out_proj": ttnn.L1_MEMORY_CONFIG,
            "attn_scores": ttnn.L1_MEMORY_CONFIG,
            "attn_softmax": ttnn.L1_MEMORY_CONFIG,
            "qkv_permute": ttnn.L1_MEMORY_CONFIG,
        },
        "norm": ttnn.L1_MEMORY_CONFIG,
        "ffn": ttnn.L1_MEMORY_CONFIG,
    },
    'ego_fut_decoder': {
        "linear1": ttnn.L1_MEMORY_CONFIG,
        "linear2": ttnn.L1_MEMORY_CONFIG,
    }
})

import math

se_matmul_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(5, 4),
    in0_block_w=1,
    out_subblock_h=1,        
    out_subblock_w=1,        
    out_block_h=math.ceil(10000 / 32 / 5),  
    out_block_w=math.ceil(256 / 32 / 4),
    per_core_M=math.ceil(10000 / 32 / 5),
    per_core_N=math.ceil(256 / 32 / 4),
    transpose_mcast=True, # transpose for col only
    fuse_batch=True,
    fused_activation=None,
)

tokenlearner_fc1_matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
    in0_block_w=16,
    out_subblock_h=1,        
    out_subblock_w=2,        
    out_block_h=math.ceil(10000 / 32 / 20),          # Must equal per_core_M
    out_block_w=2,           # Must equal per_core_N
    per_core_M=math.ceil(10000 / 32 / 20),
    per_core_N=2,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=False          
)

tokenlearner_fc2_matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
    in0_block_w=2,
    out_subblock_h=1,        
    out_subblock_w=1,        
    out_block_h=math.ceil(10000 / 32 / 20),          
    out_block_w=1,
    per_core_M=math.ceil(10000 / 32 / 20),
    per_core_N=1,
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=False          
)

program_config = MyDict({
    'se_layer': {
        "mlp_reduce": se_matmul_config,
        "mlp_expand": se_matmul_config,
    },
    'tokenlearner': {
        "mlp": {
            "fc1": tokenlearner_fc1_matmul_config,
            "fc2": tokenlearner_fc2_matmul_config,
        },
    },
})
