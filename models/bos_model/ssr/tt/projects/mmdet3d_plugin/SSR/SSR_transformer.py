import math
import logging
import torch

import ttnn
import numpy as np
import torch.nn as nn

from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.transformer import (
    build_transformer_layer_sequence,
    TransformerLayerSequence,
)
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE

from bos_metal import op, device_box

from .utils import pt2tt, ReLU
from tt.projects.configs.ops_config import MyDict


logger = logging.getLogger(__name__)


@TRANSFORMER.register_module(name="SSRPerceptionTransformer_tt")
class SSRPerceptionTransformer(BaseModule):
    counter = 0
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        encoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        map_num_vec=50,
        map_num_pts_per_vec=10,
        **kwargs,
    ):
        super(SSRPerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.rotate_center = rotate_center
        self.init_layers()

        self.spatial_shapes = ttnn.Tensor(
            data=[12, 20],
            data_type=ttnn.bfloat16,
            shape=[1, 1, 1, 2],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device_box.get(),
        ).reshape([1, 2])

    def convert_torch_embeds(
        self,
        bev_queries=None,
        bev_pos=None,
        cams_embeds=None,
        level_embeds=None,
        reference_points=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **kwargs,
    ):
        """Convert torch embeddings to ttnn embeddings."""
        if isinstance(bev_queries, torch.Tensor):
            bev_queries, bev_pos, cams_embeds, level_embeds, reference_points = pt2tt(
                [
                    bev_queries.permute(1, 0, 2),
                    bev_pos.permute(1, 0, 2),
                    cams_embeds,
                    level_embeds,
                    reference_points,
                ],
                device=device_box.get(),
                memory_config=memory_config,
            )

        self.bev_queries = bev_queries
        self.bev_pos = bev_pos
        self.cams_embeds_ = cams_embeds
        self.level_embeds_ = level_embeds

        self.encoder.reference_points = ttnn.squeeze(reference_points, -1)

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""

        self.reference_points = op.Linear(self.embed_dims, 3)
        self.map_reference_points = op.Linear(self.embed_dims, 2)

        # TODO: Fuse activation function into Linear layers
        self.can_bus_mlp = nn.Sequential(
            op.Linear(18, self.embed_dims // 2),
            ReLU(inplace=True),
            op.Linear(self.embed_dims // 2, self.embed_dims),
            ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module("norm", op.LayerNorm(self.embed_dims))

    def get_bev_features(
        self,
        mlvl_feats,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        prev_bev=None,
        memory_config=MyDict(),
        program_config=MyDict(),
        debug=False,
        **kwargs
    ):
        """
        obtain bev features.
        """

        bs = mlvl_feats.shape[0]

        # obtain rotation angle and shift with ego motion    
        delta_x = ttnn.concat([each['can_bus'][0] for each in kwargs["img_metas"]], 0)
        delta_x = ttnn.to_layout(delta_x, ttnn.Layout.TILE, memory_config=ttnn.L1_MEMORY_CONFIG)
        delta_y = ttnn.concat([each['can_bus'][1] for each in kwargs["img_metas"]], 0)
        delta_y = ttnn.to_layout(delta_y, ttnn.Layout.TILE, memory_config=ttnn.L1_MEMORY_CONFIG)
        ego_angle = ttnn.concat([each['can_bus'][-2] for each in kwargs["img_metas"]], 0)
        ego_angle = ttnn.to_layout(ego_angle, ttnn.Layout.TILE, memory_config=ttnn.L1_MEMORY_CONFIG)
        ego_angle = ttnn.div(ego_angle, math.pi)
        ego_angle = ttnn.mul(ego_angle, 180.0)
        
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = ttnn.sqrt(
            ttnn.square(delta_x) +
            ttnn.square(delta_y)
        )
        translation_angle = ttnn.div(
            ttnn.atan2(delta_y, delta_x), 
            math.pi
        )
        ttnn.deallocate(delta_x)
        ttnn.deallocate(delta_y)
        translation_angle = ttnn.mul(translation_angle, 180.0)
        bev_angle = ego_angle - translation_angle
        ttnn.deallocate(ego_angle)
        ttnn.deallocate(translation_angle)

        shift_y = ttnn.mul(translation_length, ttnn.cos(ttnn.mul(ttnn.div(bev_angle, 180.0), math.pi)))
        shift_y = ttnn.div(shift_y, grid_length_y)
        shift_y = ttnn.div(shift_y, bev_h)

        shift_x = ttnn.mul(translation_length, ttnn.sin(ttnn.mul(ttnn.div(bev_angle, 180.0), math.pi)))
        shift_x = ttnn.div(shift_x, grid_length_x)
        shift_x = ttnn.div(shift_x, bev_w)

        ttnn.deallocate(translation_length)
        ttnn.deallocate(bev_angle)

        shift_y = ttnn.unsqueeze(ttnn.unsqueeze(shift_y * self.use_shift, 0), 0)
        shift_x = ttnn.unsqueeze(ttnn.unsqueeze(shift_x * self.use_shift, 0), 0)
        shift = ttnn.permute(ttnn.concat([shift_x, shift_y], 0), (1, 0))  # xy, bs -> bs, xy
        ttnn.deallocate(shift_x)
        ttnn.deallocate(shift_y)

        if prev_bev is not None:
            if prev_bev.shape[0] == bev_h * bev_w:
                # Ensure batch first
                prev_bev = prev_bev.permute(1, 0, 2)

        # add can bus signals
        can_bus = kwargs["img_metas"][0]["can_bus"]
        can_bus = ttnn.squeeze(can_bus, 0)
        can_bus = ttnn.to_layout(can_bus, ttnn.Layout.TILE)
        can_bus = self.can_bus_mlp(can_bus)
        if self.use_can_bus:
            bev_queries = self.bev_queries + can_bus

        bs, num_cam, _, c = mlvl_feats.shape
        if self.use_cams_embeds:
            mlvl_feats = ttnn.add_(mlvl_feats, self.cams_embeds_)
        mlvl_feats = ttnn.add_(mlvl_feats, self.level_embeds_)

        bev_embed = self.encoder(
            bev_queries,
            mlvl_feats,
            mlvl_feats,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=self.bev_pos,
            spatial_shapes=self.spatial_shapes,
            prev_bev=prev_bev,
            shift=shift,
            memory_config=memory_config["encoder"],
            program_config=program_config["encoder"],
            debug=debug,
            **kwargs
        )

        return bev_embed


@TRANSFORMER_LAYER_SEQUENCE.register_module(name="CustomTransformerDecoder_tt")
class CustomTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(CustomTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        key_padding_mask=None,
        *args,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs,
            )

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query
