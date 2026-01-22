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

import torch.nn as nn
import ttnn
from tt.projects.configs.resnet50 import module_config as resnet_config
from tt.projects.configs.fpn import module_config as fpn_config
from mmdet.models import DETECTORS
from tt.projects.configs.ops_config import MyDict
from tt.projects.mmdet3d_plugin.SSR.utils import builder

from bos_metal import device_box

logger = logging.getLogger(__name__)


@DETECTORS.register_module(name="SSR_tt")
class SSR(nn.Module):
    """SSR model."""

    def __init__(
        self,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
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

        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_backbone.load_config_dict(resnet_config)
        self.img_neck = builder.build_neck(img_neck)
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
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        memory_config=MyDict(),
        program_config=MyDict(),
        **kwargs,
    ):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = ttnn.clone(img_metas[0][0]["can_bus"][:3])
        tmp_angle = ttnn.clone(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] *= 0
            img_metas[0][0]["can_bus"][:3] *= 0

        img_feats = self.extract_feat(img=img[0])
        img_feats = ttnn.reshape(
            ttnn.sharded_to_interleaved(img_feats, memory_config=ttnn.L1_MEMORY_CONFIG),
            (1, 6, 12 * 20, 256),
        )
        img_feats = ttnn.to_memory_config(img_feats, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        outs = self.pts_bbox_head(
            mlvl_feats=img_feats,
            img_metas=img_metas[0],
            prev_bev=self.prev_frame_info["prev_bev"],
            cmd=ego_fut_cmd,
            ego_his_trajs=ego_his_trajs[0],
            ego_lcf_feat=ego_lcf_feat[0],
            debug=self.debug,
            memory_config=memory_config["SSRHead"],
            program_config=program_config["SSRHead"],
        )

        # new_prev_bev, bbox_results = self.simple_test(
        #     outs=outs,
        #     img_metas=img_metas[0],
        #     ego_fut_cmd=ego_fut_cmd[0],
        #     **kwargs,
        # )

        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_bev"] = ttnn.clone(outs["bev_embed"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # memory_config=self.prev_frame_info["prev_bev"].memory_config() if self.prev_frame_info["prev_bev"] else new_prev_bev.memory_config(),
        self.prev_frame_info["prev_angle"] = tmp_angle

        return outs

    def simple_test(
        self,
        outs,
        img_metas,
        ego_fut_cmd=None,
        **kwargs,
    ):
        """Test function without augmentaiton."""
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            outs,
            ego_fut_cmd=ego_fut_cmd,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
            result_dict["metric_results"] = metric_dict

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        outs,
        ego_fut_cmd=None,
        **kwargs,
    ):
        """Test function"""
        bbox_results = []
        for i in range(len(outs["ego_fut_preds"])):
            bbox_result = dict()
            bbox_result["ego_fut_preds"] = outs["ego_fut_preds"][i]
            bbox_result["ego_fut_cmd"] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, "only support batch_size=1 now"
        metric_dict = None

        return outs["bev_embed"], bbox_results, metric_dict

    def set_epoch(self, epoch):
        self.pts_bbox_head.epoch = epoch
