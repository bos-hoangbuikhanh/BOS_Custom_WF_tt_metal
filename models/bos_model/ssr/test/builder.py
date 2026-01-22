# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry, build_from_cfg

from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS, SHARED_HEADS)

MODELS = Registry('models', parent=MMCV_MODELS)

VOXEL_ENCODERS = MODELS
MIDDLE_ENCODERS = MODELS
FUSION_LAYERS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_shared_head(cfg):
    """Build shared head of detector."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss function."""
    return LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)



# BBOX_ASSIGNERS = Registry('bbox_assigner')
# BBOX_SAMPLERS = Registry('bbox_sampler')
# BBOX_CODERS = Registry('bbox_coder')
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, BBOX_SAMPLERS, BBOX_CODERS

def build_assigner(cfg, **default_args):
    """Builder of box assigner."""
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    """Builder of box sampler."""
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder."""
    return build_from_cfg(cfg, BBOX_CODERS, default_args)