from test.utils import masked_fill

import tracy
from bos_metal import op
from mmcv.cnn.bricks.registry import ATTENTION

import ttnn
import tracy

from tt.projects.configs.ops_config import MyDict

@ATTENTION.register_module(name="TemporalSelfAttention_tt", force=True)
class TemporalSelfAttention(op.BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_bev_queue=2,
        batch_first=True,
        device=None,
        **kwargs,
    ):
        super(TemporalSelfAttention, self).__init__(device=device, **kwargs)
        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue

        self.sampling_offsets = op.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2,
        )
        self.attention_weights = op.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points,
        )
        self.value_proj = op.Linear(embed_dims, embed_dims)
        self.output_proj = op.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        bilinear_weight_hash=None,
        memory_config=MyDict(),
        program_config=MyDict(),
        **kwargs,
    ):
        tmp = ttnn.sharded_to_interleaved(query, memory_config=ttnn.L1_MEMORY_CONFIG)
        if value is None:
            assert self.batch_first, "batch_first should be True if value is None"
            value = ttnn.concat([tmp, tmp], 0, memory_config=ttnn.L1_MEMORY_CONFIG)

        if identity is None:
            identity = query
        if query_pos is not None:
            query_ = ttnn.add_(tmp, query_pos)

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape

        query = ttnn.concat([value[:bs], query_], -1)
        ttnn.deallocate(query_)

        # value = ttnn.reallocate(value)
        # NOTE: Using `ttnn.reallocate` here causes a pcc drop
        # TODO: Investigate why
        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, memory_config["value"].value)
        value = ttnn.reallocate(value)
        value_proj = self.value_proj(
            value,
            dtype=ttnn.bfloat8_b,
            memory_config=memory_config["value_proj"].value,
            program_config=program_config["value_proj"].value,
        )
        ttnn.deallocate(value)
        if key_padding_mask is not None:
            value_proj = masked_fill(value_proj, key_padding_mask[..., None], 0.0)
        value = ttnn.to_layout(value_proj, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        # value = ttnn.to_layout(value_proj, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.reshape(value, (bs * self.num_bev_queue, num_value, self.num_heads, -1))
        value = ttnn.reallocate(value)

        attention_weights = self.attention_weights(query)
        attention_weights = ttnn.reshape(
            attention_weights, (num_query * self.num_heads * self.num_bev_queue, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, -1)
        attention_weights = ttnn.reshape(
            attention_weights, (num_query * self.num_heads, self.num_bev_queue * self.num_levels * self.num_points)
        )

        sampling_offsets = self.sampling_offsets(query)
        sampling_locations = ttnn.add_(sampling_offsets, reference_points)
        sampling_locations = ttnn.reshape(
            sampling_locations, (num_query * self.num_heads, self.num_bev_queue * self.num_levels * self.num_points * 2)
        )

        if spatial_shapes.layout != ttnn.TILE_LAYOUT:
            spatial_shapes = ttnn.to_layout(spatial_shapes, ttnn.TILE_LAYOUT)
        output = ttnn.bos_ssr_deformable_attention(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
            is_denormed_grid=True,
            bilinear_weight_hash=bilinear_weight_hash,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            num_queries=num_query,
            num_levels=self.num_levels,
            num_points=self.num_points,
            is_QHB=True
        )
        ttnn.deallocate(value)

        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        output = ttnn.to_memory_config(output, identity.memory_config())
        output = self.output_proj(
            output,
            memory_config=memory_config["output_proj"].value,
            program_config=program_config["output_proj"].value,
        )
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return ttnn.add_(identity, output)
