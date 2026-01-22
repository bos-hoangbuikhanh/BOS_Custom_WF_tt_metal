import warnings
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner import BaseModule

import torch
import ttnn
from bos_metal import op
from test.utils import pt2tt, tt2pt, masked_fill


@ATTENTION.register_module(force=True)
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
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = op.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = op.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = op.Linear(embed_dims, embed_dims)

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
        level_start_index=None,
        **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

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
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = masked_fill(value, key_padding_mask[..., None], 0.0)
            
        value = ttnn.reshape(value, (bs, num_value,
                                     self.num_heads, -1))
        value = tt2pt(value, dtype=torch.float32)
        
        sampling_offsets = ttnn.reshape(
            self.sampling_offsets(query),
            (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )
        
        attention_weights = ttnn.reshape(
            self.attention_weights(query),
            (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.reshape(
            ttnn.softmax(attention_weights, -1),
            (bs, num_query, self.num_heads, self.num_levels, self.num_points),
        )

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            # bs, num_query, num_Z_anchors, xy => bs*num_query, 1, 1, num_Z_anchors*xy
            reference_points = ttnn.reshape(
                reference_points, (bs * num_query, num_Z_anchors, xy)
            )
            reference_points = ttnn.unsqueeze(reference_points, -3)

            # sampling_offsets = pt2tt(tt2pt(sampling_offsets) / \
            #     tt2pt(offset_normalizer)[None, None, None, :, None, :], device=query.device())

            bs, num_query, num_heads, num_levels, num_all_points, xy = (
                sampling_offsets.shape
            )

            sampling_offsets = ttnn.reshape(
                sampling_offsets,
                (
                    bs * num_query,
                    num_heads * num_levels * num_all_points // num_Z_anchors,
                    num_Z_anchors,
                    xy,
                ),
            )

            # TODO: Handle num_levels > 1
            sampling_offsets = ttnn.concat(
                [
                    ttnn.div(sampling_offsets[..., 0:1], offset_normalizer[0][0].item()),
                    ttnn.div(sampling_offsets[..., 1:2], offset_normalizer[0][1].item()),
                ],
                dim=-1,
            )
            # from test.utils import compare_tensors
            # compare_tensors(sampling_offsets, "sampling_offsets.pt")
            
            sampling_locations = ttnn.add_(sampling_offsets, reference_points)

            # sampling_locations = reference_points + sampling_offsets
            # sampling_locations = ttnn.reshape(sampling_locations, (bs, num_query, num_heads, num_levels, -1, num_Z_anchors, xy))
            # bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            sampling_locations = ttnn.reshape(
                sampling_locations,
                (bs, num_query, num_heads, num_levels, num_all_points, xy)
            )

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        sampling_locations = tt2pt(sampling_locations, torch.float32)
        attention_weights = tt2pt(attention_weights, torch.float32)
        output = pt2tt(
            multi_scale_deformable_attn_pytorch(
                value,
                spatial_shapes.to(dtype=torch.int32),
                sampling_locations,
                attention_weights,
            ),
            device=query.device(),
        )

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output
