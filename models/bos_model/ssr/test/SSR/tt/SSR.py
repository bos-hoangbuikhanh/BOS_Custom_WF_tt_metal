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
import time
import warnings

import torch
import torch.nn as nn
import ttnn
from tt.projects.configs.resnet50 import module_config as resnet_config
from tt.projects.configs.fpn import module_config as fpn_config
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from test.configs.op_configs import MyDict
from test.builder import build_head, build_neck, build_backbone

from bos_metal import device_box

from test.utils import pt2tt

logger = logging.getLogger(__name__)


@DETECTORS.register_module(name="SSR_tt")
class SSR(nn.Module):
    """SSR model."""

    def __init__(
        self,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        latent_world_model=None,
        train_cfg=None,
        test_cfg=None,
        video_test_mode=False,
        fut_ts=6,
        fut_mode=6,
        device=None,
        debug=False,
        **kwargs,
    ):

        super(SSR, self).__init__()
        self.device = device if device is not None else device_box.get()
        self.debug = debug

        self.pts_bbox_head = build_head(pts_bbox_head)
        self.img_backbone = build_backbone(img_backbone)
        self.img_backbone.load_config_dict(resnet_config)
        self.img_neck = build_neck(img_neck)
        self.img_neck.load_config_dict(fpn_config)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head["valid_fut_ts"]

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        self.planning_metric = None
        self.embed_dims = 256

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, "img_neck") and self.img_neck is not None

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features of images."""
        img_feats = self.img_backbone(img)
        img_feats = self.img_neck.forward(img_feats)

        return img_feats[0]

    def forward(
        self,
        img_metas,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        memory_config=MyDict(),
        program_config=MyDict(),
        **kwargs,
    ):

        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        # else:
        #     img_metas[0][0]["can_bus"][-1] = 0
        #     img_metas[0][0]["can_bus"][:3] = 0

        # ttnn.synchronize_device(device_box.get())
        img_backbone_st = time.time()
        img_feats = self.extract_feat(img=img[0])
        img_feats = ttnn.reshape(
            ttnn.sharded_to_interleaved(img_feats, memory_config=ttnn.L1_MEMORY_CONFIG),
            (1, 6, 12 * 20, 256),
        )
        img_feats = ttnn.to_memory_config(img_feats, ttnn.DRAM_MEMORY_CONFIG)
        # ttnn.synchronize_device(device_box.get())
        logger.info(f"Img backbone time: {time.time() - img_backbone_st:.4f}s")

        outs = self.pts_bbox_head(
            mlvl_feats=img_feats,
            img_metas=img_metas[0],
            prev_bev=self.prev_frame_info["prev_bev"],
            cmd=ego_fut_cmd,
            debug=self.debug,
            memory_config=memory_config["SSRHead"],
            program_config=program_config["SSRHead"],
        )

        return outs


