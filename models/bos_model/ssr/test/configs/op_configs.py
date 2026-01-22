from bos_metal import op

import ttnn

L1_HEIGHT = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
L1_WIDTH = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
L1_BLOCK = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
L1_INTERLEAVE = ttnn.L1_MEMORY_CONFIG
DRAM_INTERLEAVE = ttnn.DRAM_MEMORY_CONFIG


import math
from collections.abc import MutableMapping


class MyDict(MutableMapping):
    """
    A dictionary-like container that

    * never raises KeyError on chained lookup – missing items are auto-created
      as empty MyDict placeholders;
    * can be constructed from ordinary (possibly nested) dicts;
    * wraps every stored value so that `node.value` returns the real payload
      (or ``None`` for placeholders created implicitly).
    """

    # --------------------------------------------------------------------- #
    # core construction helpers                                             #
    # --------------------------------------------------------------------- #
    __slots__ = ("_data", "_has_payload", "_payload")

    def __init__(self, initial=None, *, _has_payload=False, _payload=None):
        self._data: dict[str, "MyDict"] = {}
        self._has_payload: bool = _has_payload
        self._payload = _payload

        # If the user passed a plain mapping, recursively convert its leaves
        if isinstance(initial, dict):
            for k, v in initial.items():
                self[k] = v  # delegates to our __setitem__
        elif initial is not None:
            # Treat any non-dict object as a leaf payload
            self._has_payload = True
            self._payload = initial

    # --------------------------------------------------------------------- #
    # MutableMapping protocol                                               #
    # --------------------------------------------------------------------- #
    def __getitem__(self, key):
        if self._has_payload:
            # You’re trying to index into a leaf node.  Instead of failing,
            # promote a child placeholder so that the chain can continue.
            self._has_payload = False
            self._payload = None

        if key not in self._data:
            # Create and store a *placeholder* that reports .value == None
            self._data[key] = MyDict()
        return self._data[key]

    def __setitem__(self, key, value):
        # Any plain dict becomes a nested MyDict; everything else
        # (numbers, lists, custom objects, …) becomes an **owned payload**
        if isinstance(value, dict) and not isinstance(value, MyDict):
            wrapped = MyDict(value)
        elif isinstance(value, MyDict):
            wrapped = value
        else:
            wrapped = MyDict(_has_payload=True, _payload=value)

        self._data[key] = wrapped

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    # --------------------------------------------------------------------- #
    # Convenience helpers                                                   #
    # --------------------------------------------------------------------- #
    @property
    def value(self):
        """
        * ``None``  → this node was auto-created during a missing-key lookup.
        * anything else → the real object stored at this leaf.
        """
        return self._payload if self._has_payload else None

    # Pretty representation (helps with debugging / printing)
    def __repr__(self):
        if self._has_payload:
            return f"MyDict(value={self._payload!r})"
        return f"MyDict({self._data!r})"


program_config = MyDict(
    {
        "SSRHead": MyDict({
                "transformer": MyDict({
                        "encoder": MyDict({
                            "self_attn": MyDict({
                                "value_proj": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                                    compute_with_storage_grid_size=(5, 4),
                                    in0_block_w=8,
                                    out_subblock_h=1,
                                    out_subblock_w=1,
                                    per_core_M=32,
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
                                    per_core_M=16,
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
                                    per_core_M=16,
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
                                    per_core_M=16,
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
                                    per_core_M=16,
                                    per_core_N=8,
                                    fuse_batch=True,
                                    fused_activation=None,
                                    mcast_in0=False,
                                ),
                            })
                        }),
                }),
                "tokenlearner": MyDict({
                        "mlp": MyDict({
                            "fc1": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                                compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
                                in0_block_w=16,
                                out_subblock_h=4,
                                out_subblock_w=2,
                                out_block_h=math.ceil(10000 / 32 / 20),
                                out_block_w=2,
                                per_core_M=16,
                                per_core_N=2,
                                fuse_batch=True,
                                fused_activation=ttnn.UnaryOpType.GELU,
                                mcast_in0=False,
                            ),
                            "fc2": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                                compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
                                in0_block_w=2,
                                out_subblock_h=4,
                                out_subblock_w=1,
                                out_block_h=math.ceil(10000 / 32 / 20),
                                out_block_w=1,
                                per_core_M=math.ceil(10000 / 32 / 20),
                                per_core_N=1,
                                fuse_batch=True,
                                fused_activation=None,
                                mcast_in0=False,
                            ),
                        }),
                    }),
            }),
    }
)

memory_config = MyDict(
    {
        "SSRHead": MyDict(
            {
                "transformer": MyDict(
                    {
                        "encoder": MyDict({
                            "bev_query": op.ShardedMemConfig(
                                shape=(512, 256),
                                core_grid=(5, 4),
                                strategy='height',
                                as_shard_shape=True
                            ),
                            "self_attn": MyDict({
                                "value": op.ShardedMemConfig((1024, 256), (5, 4), "height", as_shard_shape=True),
                                "value_proj": DRAM_INTERLEAVE,
                                "output_proj": L1_HEIGHT,
                            }),
                            "cross_attn": MyDict({
                                "output_proj": L1_HEIGHT
                            }),
                            "ffn": L1_HEIGHT,
                        })
                    }
                ),
                "navi_se": MyDict(
                    {
                        "x_se": ttnn.L1_MEMORY_CONFIG,
                        "x": op.ShardedMemConfig(
                            (64, 2016),
                            (5, 4),
                            "block",
                            orientation=ttnn.ShardOrientation.COL_MAJOR,
                            as_shard_shape=True,
                        ),
                        "mlp_reduce": ttnn.L1_MEMORY_CONFIG,
                        "mlp_expand": ttnn.L1_MEMORY_CONFIG,
                    }
                ),
                "concat": MyDict(
                    {
                        "bev_pos": op.ShardedMemConfig(
                            shape=(512, 256),
                            core_grid=(5, 4),
                            strategy='height',
                            as_shard_shape=True
                        ),
                        # "bev_navi_embed": ttnn.L1_MEMORY_CONFIG,
                        "bev_navi_embed": op.ShardedMemConfig(
                            shape=(512, 256),
                            core_grid=(5, 4),
                            strategy='height',
                            as_shard_shape=True
                        ),
                        "out_concat": op.ShardedMemConfig(
                            shape=(512, 512),
                            core_grid=(5, 4),
                            strategy='height',
                            as_shard_shape=True
                        ),
                    }
                ),
                "tokenlearner": MyDict(
                    {
                        "mlp": MyDict(
                            {
                                "input": op.ShardedMemConfig((512, 512), (5, 4), "height", as_shard_shape=True),
                                "fc1": op.ShardedMemConfig((512, 64), (5, 4), "height", as_shard_shape=True),
                                "fc2": op.ShardedMemConfig((512, 32), (5, 4), "height", as_shard_shape=True),
                            }
                        ),
                    }
                ),
                "latent_decoder": MyDict(
                    {
                        "self_attn": MyDict(
                            {
                                "q_input": ttnn.L1_MEMORY_CONFIG,
                                "kv_input": ttnn.L1_MEMORY_CONFIG,
                                "in_proj_q": ttnn.L1_MEMORY_CONFIG,
                                "in_proj_kv": ttnn.L1_MEMORY_CONFIG,
                                "qkv_permute": ttnn.L1_MEMORY_CONFIG,
                                "qkv_split": ttnn.L1_MEMORY_CONFIG,
                            }
                        ),
                        "norm": ttnn.L1_MEMORY_CONFIG,
                        "ffn": ttnn.L1_MEMORY_CONFIG,
                    }
                ),
                "way_decoder": MyDict(
                    {
                        "way_point": ttnn.L1_MEMORY_CONFIG,
                        "way_pos": ttnn.L1_MEMORY_CONFIG,
                        "cross_attn": MyDict(
                            {
                                "q_input": ttnn.L1_MEMORY_CONFIG,
                                "kv_input": ttnn.L1_MEMORY_CONFIG,
                                "in_proj_q": ttnn.L1_MEMORY_CONFIG,
                                "in_proj_kv": ttnn.L1_MEMORY_CONFIG,
                                "qkv_split": ttnn.L1_MEMORY_CONFIG,
                                "out_proj": ttnn.L1_MEMORY_CONFIG,
                                "attn_scores": ttnn.L1_MEMORY_CONFIG,
                                "attn_softmax": ttnn.L1_MEMORY_CONFIG,
                                "qkv_permute": ttnn.L1_MEMORY_CONFIG,
                            }
                        ),
                        "norm": ttnn.L1_MEMORY_CONFIG,
                        "ffn": ttnn.L1_MEMORY_CONFIG,
                    }
                ),
            }
        ),
    }
)

