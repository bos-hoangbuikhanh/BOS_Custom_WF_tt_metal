import copy
import os
import warnings

import torch
from bos_metal import engine, op
from loguru import logger
from mmcv.runner import load_checkpoint
from mmcv.utils import build_from_cfg
from mmdet.datasets import DATASETS
from mmcv.cnn.bricks.registry import ATTENTION, NORM_LAYERS
from tt.projects.configs.ops_config import memory_config, program_config
from reference.projects.mmdet3d_plugin.datasets.builder import build_dataloader
from tt.projects.mmdet3d_plugin.SSR.utils.builder import build_model
from tt.projects.mmdet3d_plugin.SSR.utils.misc import extract_data_from_container


def build_dataset(cfg, default_args=None):
    """Build dataset from cfg."""
    return build_from_cfg(cfg, DATASETS, default_args)


def build_dataloader_from_cfg(cfg, samples_per_gpu):
    """Create dataset and dataloader for testing."""
    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    return dataset, dataloader


def build_models_from_cfg(cfg, checkpoint_path=None, fuse_bn=False):
    """Construct torch and ttnn models and load checkpoint."""
    ttnn_model = build_model(cfg.tt_model, test_cfg=cfg.get("test_cfg"))

    if checkpoint_path is not None and False:
        checkpoint = load_checkpoint(ttnn_model, checkpoint_path, map_location="cpu")

        if "CLASSES" in checkpoint.get("meta", {}):
            classes = checkpoint["meta"]["CLASSES"]
            ttnn_model.CLASSES = classes
        if "PALETTE" in checkpoint.get("meta", {}):
            palette = checkpoint["meta"]["PALETTE"]
            ttnn_model.PALETTE = palette
    else:
        classes = [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ]
        ttnn_model.CLASSES = classes

    return ttnn_model


def prepare_ttnn_model(
    ttnn_model,
    input_config,
    data_loader,
    embeddings_path=None,
    checkpoint_path=None,
    torch_model=None,
    debug=False,
    device=None,
):
    """Prepare TTNN model by loading weights and warming up."""
    data_for_cache = extract_data_from_container(
        next(iter(data_loader)), tensor="tt", device=device, input_config=input_config
    )

    if os.path.exists(checkpoint_path):
        print(f"Load checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    else:
        logger.warning(f"Unable to find processed state dict from {checkpoint_path}, trying to create a new one.")
        processor = engine.ModelProcessor(torch_model)  # NOTE: This will raise an error
        state_dict = processor.process_state_dict(**data_for_cache, save_to="ssr_state_dict.pth")
    ttnn_model.load_state_dict(state_dict, strict=False)

    embed_dict = torch.load(embeddings_path, map_location=torch.device("cpu")) if embeddings_path else None
    ttnn_model.pts_bbox_head.transformer.convert_torch_embeds(**embed_dict)
    ttnn_model.pts_bbox_head.convert_torch_embeds(**embed_dict)

    ttnn_model.eval()
    ttnn_model(
        rescale=True,
        **data_for_cache,
        memory_config=memory_config,
        program_config=program_config,
    )

    ttnn_model.prev_frame_info["prev_bev"] = None

    return None


def unset_env_vars():
    """Unset environment variables that may interfere with TTNN."""

    os.unsetenv("TTNN_ENABLE_LOGGING")
    os.unsetenv("ENABLE_PROFILER")
    os.unsetenv("TT_METAL_DPRINT_CORES")
    os.unsetenv("TT_METAL_DPRINT_CHIPS")
    os.unsetenv("ENABLE_TRACY")
    os.unsetenv("TT_METAL_DEVICE_PROFILER")
    os.unsetenv("TT_METAL_PROFILER_SYNC")
    os.unsetenv("TT_METAL_DEVICE_PROFILER_DISPATCH")

# Register sub-modules
def register_ttnn_submodules():
    if "LN_tt" not in NORM_LAYERS.module_dict:
        NORM_LAYERS.register_module("LN_tt", module=op.LayerNorm, force=True)
    if "MultiheadAttention_tt" not in ATTENTION.module_dict:
        ATTENTION.register_module("MultiheadAttention_tt", module=op.MultiheadAttention, force=True)
