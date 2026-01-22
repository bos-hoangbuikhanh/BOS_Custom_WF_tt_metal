"""
 Copyright (c) Zhijia Technology. All rights reserved.
 
 Author: Peidong Li (lipeidong@smartxtruck.com / peidongl@outlook.com)
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 
 Modifications:
 - 2025-08-11: Refactored to use ttnn for tensor operations - Nhat Nguyen (nhatnguyen@bos-semi.com)
"""

import copy
import logging

import torch
import torch.nn as nn
from mmcv.cnn.bricks import build_activation_layer
from mmcv.cnn.bricks.transformer import (
    build_positional_encoding,
    build_transformer_layer_sequence,
)
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import build_transformer

from bos_metal import device_box, op
from test.builder import build_bbox_coder
from test.configs.op_configs import MyDict
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
import ttnn

# from .tokenlearner import *
from test.SSR.tt.tokenlearner import TokenLearnerV11
from test.SSR.tt.bbox_coder import fut_nms_free_coder, map_nms_free_coder, nms_free_coder
from bos_metal import op, device_box
from test.utils import tt2pt, pt2tt, compare_tensors, pad_to_multiple, ReLU, Sigmoid

logger = logging.getLogger(__name__)


class SELayer(op.BaseModule):
    def __init__(self, channels, act_layer=ReLU, gate_layer=Sigmoid):
        super().__init__()
        self.mlp_reduce = op.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = op.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se, inplace=False, memory_config=MyDict(), program_config=MyDict()):
        x_se = self.mlp_reduce(
            x_se,
            memory_config=memory_config["mlp_reduce"].value,
            program_config=program_config["mlp_reduce"].value,
            # activation='relu'
        )
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(
            x_se,
            memory_config=memory_config["mlp_expand"].value,
            program_config=program_config["mlp_expand"].value,
        )
        
        if inplace:
            return ttnn.multiply_(x, self.gate(x_se))
        else:
            return ttnn.multiply(x, self.gate(x_se), memory_config=x.memory_config())
    
@HEADS.register_module(name="SSRHead_tt")
class SSRHead(nn.Module):
    """Head of SSR model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(
        self,
        *args,
        transformer=None,
        bbox_coder=None,
        bev_h=30,
        bev_w=30,
        fut_ts=6,
        fut_mode=6,
        map_num_vec=20,
        map_num_pts_per_vec=2,
        map_query_embed_type="all_pts",
        num_scenes=16,
        latent_decoder=None,
        way_decoder=None,
        ego_fut_mode=3,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        num_classes,
        in_channels,
        num_query=100,
        num_reg_fcs=2,
        positional_encoding=dict(
            type="SinePositionalEncoding",
            num_feats=128,
            normalize=True,
        ),
        test_cfg=dict(max_per_img=100),
        device=None,
        **kwargs,
    ):

        super(SSRHead, self).__init__()
        self.device = device if device is not None else device_box.get()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode

        self.latent_decoder = latent_decoder
        self.way_decoder = way_decoder

        self.ego_fut_mode = ego_fut_mode
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.num_scenes = num_scenes

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.map_query_embed_type = map_query_embed_type
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.act_cfg = transformer.get("act_cfg", dict(type="ReLU_tt", inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        ego_fut_decoder = []
        ego_fut_dec_in_dim = (
            self.embed_dims + len(self.ego_lcf_feat_idx) if self.ego_lcf_feat_idx is not None else self.embed_dims
        )
        for _ in range(self.num_reg_fcs):
            # TODO: Fuse Linear and ReLU
            ego_fut_decoder.append(op.Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(ReLU())
        ego_fut_decoder.append(op.Linear(ego_fut_dec_in_dim, 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        self.navi_se = SELayer(self.embed_dims)

        self.way_point = op.Embedding(self.ego_fut_mode * self.fut_ts, self.embed_dims * 2)
        self.tokenlearner = TokenLearnerV11(self.num_scenes, self.embed_dims * 2)

        self.latent_decoder = build_transformer_layer_sequence(self.latent_decoder)
        self.way_decoder = build_transformer_layer_sequence(self.way_decoder)

    def convert_torch_embeds(
        self,
        way_point,
        wp_pos,
        navi_embd,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **kwargs,
    ):
        """Convert torch embedding to tt embedding."""
        if isinstance(way_point, torch.Tensor):
            way_point, wp_pos, navi_embd = pt2tt(
                [way_point.unsqueeze(0), wp_pos.unsqueeze(0), navi_embd],
                device=self.device,
                memory_config=memory_config,
            )

        self.way_point_ = way_point
        self.wp_pos = wp_pos
        self.navi_embd = navi_embd

    def forward(
        self,
        mlvl_feats,
        img_metas,
        prev_bev=None,
        only_bev=False,
        cmd=None,
        memory_config=MyDict(),
        program_config=MyDict(),
        debug=False,
        **kwargs,
    ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _ = mlvl_feats.shape

        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            img_metas=img_metas,
            prev_bev=prev_bev,
            memory_config=memory_config["transformer"],
            program_config=program_config["transformer"],
            debug=debug,
        )
        ttnn.deallocate(mlvl_feats)

        if only_bev:
            return bev_embed

        cmd = cmd[0][0, 0, 0]
        for cmd_idx, cmd_ in enumerate(cmd):
            if cmd_ != 0:
                break

        x_se = ttnn.to_memory_config(
            self.navi_embd[cmd_idx : cmd_idx + 1], 
            memory_config=memory_config["navi_se"]["x_se"].value
        )
        bev_embed_out = ttnn.to_memory_config(bev_embed, ttnn.DRAM_MEMORY_CONFIG)
        bev_navi_embed = self.navi_se(
            bev_embed, x_se,
            memory_config=memory_config["navi_se"],
            program_config=program_config["navi_se"],
            inplace=True
        )
        ttnn.deallocate(x_se)

        # bev_pos = ttnn.to_memory_config(
        #     ttnn.to_layout(self.transformer.bev_pos, ttnn.ROW_MAJOR_LAYOUT),
        #     memory_config=memory_config["concat"]["bev_pos"].value
        # )
        bev_navi_embed = ttnn.unsqueeze(bev_navi_embed, 0)
        # TODO: Concat causes hanging with 2 HEIGHT sharded tensors (with TILE layout)
        bev_query = ttnn.concat(
            [
                ttnn.sharded_to_interleaved(bev_navi_embed, memory_config=ttnn.L1_MEMORY_CONFIG),
                ttnn.unsqueeze(ttnn.to_memory_config(self.transformer.bev_pos, ttnn.L1_MEMORY_CONFIG), 0)
            ], dim=-1, 
            # memory_config=memory_config["concat"]["out_concat"].value
        )
        ttnn.deallocate(bev_navi_embed)
        # bev_query = ttnn.sharded_to_interleaved(bev_query, memory_config=ttnn.L1_MEMORY_CONFIG)
        # bev_query = ttnn.sharded_to_interleaved(bev_query)
        # bev_query = ttnn.reallocate(bev_query, bev_query.memory_config())

        learned_latent_query = self.tokenlearner(
            bev_query,
            memory_config=memory_config["tokenlearner"],
            program_config=program_config["tokenlearner"]
        )
        ttnn.deallocate(bev_query)

        latent_query, latent_pos = (
            learned_latent_query[:, :, : self.embed_dims],
            learned_latent_query[:, :, self.embed_dims :],
        )
        ttnn.deallocate(learned_latent_query)

        latent_query = self.latent_decoder(
            query=latent_query,
            key=latent_query,
            value=latent_query,
            query_pos=latent_pos,
            key_pos=latent_pos,
            memory_config=memory_config["latent_decoder"],
            program_config=program_config["latent_decoder"],
            inplace=True,
        )

        way_point_tt = ttnn.to_memory_config(
            self.way_point_, memory_config=memory_config["way_decoder"]["way_point"].value
        )
        wp_pos_tt = ttnn.to_memory_config(self.wp_pos, memory_config=memory_config["way_decoder"]["way_pos"].value)

        way_point = self.way_decoder(
            query=way_point_tt,
            key=latent_query,
            value=latent_query,
            query_pos=wp_pos_tt,
            key_pos=latent_pos,
            memory_config=memory_config["way_decoder"],
            program_config=program_config["way_decoder"],
            inplace=True,
        )

        outputs_ego_trajs = self.ego_fut_decoder(way_point)
        outputs_ego_trajs = ttnn.reshape(outputs_ego_trajs, (1, self.ego_fut_mode, self.fut_ts, 2))
        # outputs_ego_trajs_fut = outputs_ego_trajs[:, cmd_idx : cmd_idx + 1]

        ttnn.deallocate(way_point_tt)
        ttnn.deallocate(wp_pos_tt)

        outs = {
            "bev_embed": bev_embed_out,
            # "scene_query": tt2pt(latent_query, torch.float32).as_subclass(torch.Tensor),
            # 'act_query': act_query,
            # 'act_pos': act_pos,
            "ego_fut_preds": outputs_ego_trajs,
        }

        return outs

