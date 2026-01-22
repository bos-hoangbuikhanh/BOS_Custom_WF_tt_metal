import tracy

import torch
import ttnn
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.runner.base_module import BaseModule

from bos_metal import device_box, op

from ..utils.misc import masked_fill, pt2tt
from tt.projects.configs.ops_config import MyDict


@ATTENTION.register_module(name="SpatialCrossAttention_tt")
class SpatialCrossAttention(op.BaseModule):
    count = 0
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(
        self,
        embed_dims=256,
        num_cams=6,
        pc_range=None,
        init_cfg=None,
        batch_first=False,
        deformable_attention=dict(type="MSDeformableAttention3D", embed_dims=256, num_levels=4),
        **kwargs,
    ):
        super(SpatialCrossAttention, self).__init__()

        self.init_cfg = init_cfg
        self.dropout = op.Identity()
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = op.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.max_len = [3_072, 1_344, 1_376, 3_680, 928, 992]
        self.slots_ = ttnn.zeros((1, 10000, embed_dims), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device_box.get())


    def forward(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        spatial_shapes=None,
        reference_points_rebatch=None,
        indexes=None,
        count=None,
        bilinear_weight_hash=None,
        memory_config=MyDict(),
        program_config=MyDict(),
        **kwargs,
    ):

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        query = ttnn.sharded_to_interleaved(query, memory_config=ttnn.L1_MEMORY_CONFIG)
        query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
        query = ttnn.reallocate(query)

        groups = [[0], [1, 2], [3], [4, 5]]
        queries = []

        for group_idx, group in enumerate(groups):
            i = group[0]
            if len(group) == 1:
                q_group = ttnn.to_layout(ttnn.bos_getitem(query, [indexes[i]], [1]), ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            else:
                tmp = [ttnn.to_layout(ttnn.bos_getitem(query, [indexes[j]], [1]), ttnn.TILE_LAYOUT) for j in group]
                q_group = ttnn.concat(tmp, dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)
                [ttnn.deallocate(t) for t in tmp]
                q_group = ttnn.reallocate(q_group)
            ref = reference_points_rebatch[i]
            out = self.deformable_attention(
                query=q_group,
                value=value[group_idx],
                reference_points=ref,
                spatial_shapes=spatial_shapes,
                bilinear_weight_hash=bilinear_weight_hash
            )
            queries.append(out)
        ttnn.deallocate(query)
        
        slots = ttnn.clone(self.slots_, memory_config=ttnn.L1_MEMORY_CONFIG)
        for group_idx, group in enumerate(groups):
            q = queries[group_idx]
            for i in group:
                if len(group) > 1:
                    q_i = q[group.index(i):group.index(i)+1]
                else:
                    q_i = q
                tmp = ttnn.bos_getitem(slots, [indexes[i]], [1])
                tmp = ttnn.to_layout(tmp, ttnn.TILE_LAYOUT)
                tmp = ttnn.add_(tmp, q_i)
                tmp = ttnn.to_layout(tmp, ttnn.ROW_MAJOR_LAYOUT)
                slots[:, indexes[i]] = tmp
                ttnn.deallocate(tmp)
            ttnn.deallocate(q_i)
            ttnn.deallocate(q)

        slots = ttnn.div(slots, count)
        slots = ttnn.to_layout(slots, ttnn.TILE_LAYOUT)
        slots = ttnn.to_memory_config(slots, inp_residual.memory_config())
        slots = self.output_proj(
            slots, 
            memory_config=memory_config["output_proj"].value,
            program_config=program_config["output_proj"].value
        )

        SpatialCrossAttention.count += 1
        return ttnn.add_(inp_residual, slots)


@ATTENTION.register_module()
class MSDeformableAttention3D_tt(BaseModule):
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
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = op.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = op.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = op.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        bilinear_weight_hash=None,
        **kwargs,
    ):

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))
        bs, num_query, _ = query.shape
        num_value = value.shape[-2]

        bs, num_query, _ = reference_points.shape

        attention_weights = self.attention_weights(query, memory_config=ttnn.L1_MEMORY_CONFIG)
        attention_weights = ttnn.reshape(
            attention_weights, (bs * num_query * self.num_heads, self.num_levels * self.num_points)
        )
        query = ttnn.reallocate(query)
        attention_weights = ttnn.reallocate(attention_weights)
        attention_weights = ttnn.softmax(attention_weights, -1)
        attention_weights = ttnn.reallocate(attention_weights)

        sampling_offsets = self.sampling_offsets(query, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(query)
        sampling_offsets_ = ttnn.add_(sampling_offsets, reference_points)
        sampling_locations = ttnn.reshape(
            sampling_offsets_, (bs * num_query * self.num_heads, self.num_levels * self.num_points * 2), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(sampling_offsets_)

        if spatial_shapes.layout != ttnn.TILE_LAYOUT:
            spatial_shapes = ttnn.to_layout(spatial_shapes, ttnn.TILE_LAYOUT)

        value = self.value_proj(value, memory_config=ttnn.L1_MEMORY_CONFIG)
        if key_padding_mask is not None:
            value = masked_fill(value, key_padding_mask[..., None], 0.0)
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))
        
        output = ttnn.bos_ssr_deformable_attention(
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
            is_denormed_grid=True,
            bilinear_weight_hash=bilinear_weight_hash,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            use_fp32=False,
            num_queries=num_query,
            num_levels=self.num_levels,
            num_points=self.num_points,
            is_QHB=False
        )
        ttnn.deallocate(value)
        ttnn.deallocate(attention_weights)
        ttnn.deallocate(sampling_locations)
        output = ttnn.reallocate(output)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output
