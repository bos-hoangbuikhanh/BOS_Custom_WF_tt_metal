import tracy
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention

from bos_metal import op, device_box
import ttnn
from test.SSR.tt.ms_deformable_attention_3d import MSDeformableAttention3D_tt

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
        max_len=3_680,
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
        # tracy.signpost("SpatialCrossAttention")
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

        # get queries for each image 
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
                ttnn.deallocate(q_i)
                tmp = ttnn.to_layout(tmp, ttnn.ROW_MAJOR_LAYOUT)
                slots[:, indexes[i]] = tmp
            ttnn.deallocate(tmp)
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
