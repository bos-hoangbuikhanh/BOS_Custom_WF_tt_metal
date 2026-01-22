import pytest
import os
import time
import sys
import warnings
from copy import deepcopy
from typing import List
import os.path as osp
import sys
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from test.builder import build_model

# Force import to register modules (side effects for registries)
from test.SSR.reference import SSR as _ref_SSR
from test.SSR.reference import SSR_head as _ref_SSR_head  # noqa: F401
from test.SSR.reference import SSR_transformer as _ref_SSR_transformer
from test.SSR.reference import encoder as _ref_encoder
from test.SSR.reference import tokenlearner as _ref_tokenlearner
from test.SSR.tt import SSR as _tt_SSR
from test.SSR.tt import SSR_head as _tt_SSR_head  # noqa: F401
from test.SSR.tt import SSR_transformer as _tt_SSR_transformer
from test.SSR.tt import tokenlearner as _tt_tokenlearner
from test.SSR.tt import encoder as _tt_encoder
from test.SSR.tt import fpn as _tt_fpn
from test.SSR.tt import positional_encoding as _tt_posenc
from test.SSR.tt import resnet as _tt_resnet
from test.SSR.tt import spatial_cross_attention as _tt_spatial_cross_attention
from test.SSR.tt import temporal_self_attention as _tt_temporal_self_attention
from test.SSR.tt import transformer as _tt_transformer
from test.SSR.tt.utils.misc import extract_data_from_container
from test.utils import tt2pt
from typing import List

import numpy as np
import torch

# Project-specific imports
from bos_metal import compare_tensors, device_box, op
from loguru import logger
from mmcv import Config
from mmcv.cnn.bricks.registry import ATTENTION, NORM_LAYERS
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint
from mmcv.parallel import DataContainer
from mmdet.models.builder import BACKBONES, NECKS  # kept for side-effect registrations
from tt.projects.configs.ops_config import memory_config, program_config
from tt.projects.configs.resnet50 import module_config


import ttnn

# ------------------------------------------------------------------------------
# Global setup
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

# If your repo needs a dynamic path, replace this with WORKING_DIR or remove it.
sys.path.append("")  # ⚠️ Prefer an absolute path or environment-based path

# Register custom ops
NORM_LAYERS.register_module("LN_tt", module=op.LayerNorm)
ATTENTION.register_module("MultiheadAttention_tt", module=op.MultiheadAttention)

MEMORY_ALIGNMENT = 16


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
BASE_DIR = os.environ.get("WORKING_DIR", None)
assert BASE_DIR, "WORKING_DIR should not be None"
BASE_DIR = Path(BASE_DIR).resolve()

COMMON_CONFIG_PATH = BASE_DIR / "test" / "configs" / "model_configs.py"
EMBED_PATH = BASE_DIR / "data" / "embeddings" / "tensor_dict.pth"
PT_CHECKPOINT_PATH = BASE_DIR / "data" / "ckpts" / "ssr_pt.pth"
TT_CHECKPOINT_PATH = BASE_DIR / "data" / "ckpts" / "ssr_tt.pth"

COMMON_CONFIG_PATH = str(COMMON_CONFIG_PATH)
EMBED_PATH = str(EMBED_PATH)
PT_CHECKPOINT_PATH = str(PT_CHECKPOINT_PATH)
TT_CHECKPOINT_PATH = str(TT_CHECKPOINT_PATH)

NUM_SAMPLES = 2
# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Build “safe” metadata (avoid geometric NaNs)
# ------------------------------------------------------------------------------
def setup_dram_sharded_input(ttnn_input, device):
    def divup(a, b):
        return (a + b - 1) // b

    dram_grid_size = device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(ttnn_input.volume() // ttnn_input.padded_shape[-1], (dram_grid_size.x * dram_grid_size.y)),
            ttnn_input.padded_shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config_DRAM = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )
    return ttnn_input, sharded_mem_config_DRAM


def make_pinhole_lidar2img_list(num_cams=6, H=384, W=640, fx=500.0, fy=500.0, z_cam=5.0):
    """Generate 6 lidar2img with valid intrinsics and Tz>0 to avoid zero division when projecting."""
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx, 0.0], [0, fy, cy, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1.0]], dtype=np.float32)
    outs_np = []
    yaws = np.linspace(-np.pi, np.pi, num_cams, endpoint=False)
    for yaw in yaws:
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, z_cam], [0, 0, 0, 1]], dtype=np.float32)
        lidar2cam = T @ R
        outs_np.append((K @ lidar2cam).astype(np.float32))

    return outs_np


def make_img_metas(num_cams=6, img_shape=(384, 640, 3)):
    """Create img_metas"""
    can_bus = np.zeros(18, dtype=np.float32)
    lidar2img_np = make_pinhole_lidar2img_list(num_cams, H=img_shape[0], W=img_shape[1])
    meta_np = {
        "can_bus": can_bus.astype(np.float32),
        "lidar2img": lidar2img_np,
        "img_shape": [img_shape] * num_cams,
        "pad_shape": [img_shape] * num_cams,
        "ori_shape": [img_shape] * num_cams,
    }

    return [[meta_np]]


def convert_images_to_tt(batch_nchw: torch.Tensor) -> List[ttnn.Tensor]:
    """
    Convert PyTorch images from NCHW to a list of TTNN tensors prepared for the pipeline:
    - Permute to NHWC
    - Row-major TT tensor
    - Reshape to (1, 1, H*W*C, C)
    - Pad channel dim to 16 (requirement for some kernels)
    """
    assert batch_nchw.ndim == 4, f"Expected 4D tensor, got {batch_nchw.shape}"
    nhwc = batch_nchw.permute(0, 2, 3, 1)
    out: List[ttnn.Tensor] = []

    for i in range(nhwc.shape[0]):
        x = ttnn.from_torch(nhwc[i : i + 1], dtype=ttnn.bfloat16, layout=ttnn.Layout.ROW_MAJOR)
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
        # pad last dim to 16
        x = ttnn.pad(x, [x.shape[0], x.shape[1], x.shape[2], 16], [0, 0, 0, 0], 0)
        out.append(x)

    return out


def make_inputs(cfg, seed: int = 0):
    img_config = cfg.input_config["img_config"]
    BATCH_SIZE = img_config["batch_size"]
    NUM_CAMS = img_config["num_cams"]
    C_IN = img_config["img_channels"]
    IMG_H = img_config["img_height"]
    IMG_W = img_config["img_width"]

    torch.manual_seed(seed)
    ref_img = [torch.rand(BATCH_SIZE, NUM_CAMS, C_IN, IMG_H, IMG_W)]
    img_metas_np = make_img_metas(num_cams=NUM_CAMS, img_shape=(IMG_H, IMG_W, C_IN))
    ego_fut_cmd = [torch.tensor([[[[0.0, 1.0, 0.0]]]], dtype=torch.float32)]

    return ref_img, img_metas_np, ego_fut_cmd


class MockDataLoader:
    def __init__(self, cfg, num_samples=2, seed=0):
        self.cfg = cfg
        self.num_samples = num_samples
        self.dataset = list(range(num_samples))  # mimic __len__
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self.num_samples:
            raise StopIteration

        ref_img, img_metas_np, ego_fut_cmd = make_inputs(cfg=self.cfg, seed=self.seed + self._index)

        data = {  # list of DCs
            "img": [DataContainer(ref_img)],
            "img_metas": [DataContainer(img_metas_np)],
            "ego_fut_cmd": [DataContainer(ego_fut_cmd)],
        }

        self._index += 1

        return data


@contextmanager
def tt_forward(model, **kwargs):
    """
    Context manager to run TT forward and auto-deallocate TT tensors on exit.
    Yields the raw outputs for immediate use.
    """
    outputs = model(**kwargs)
    try:
        yield outputs
    finally:
        # Deallocate TTNN tensors in outputs (dict or list)
        if isinstance(outputs, dict):
            it = outputs.values()
        elif isinstance(outputs, (list, tuple)):
            it = outputs
        else:
            it = []
        for v in it:
            if isinstance(v, ttnn.Tensor):
                ttnn.deallocate(v)


def load_embed_dict(path: str):
    if not path or not osp.exists(path):
        logger.warning("Embedding path missing; skipping embed conversion: %s", path)
        return None
    return torch.load(path, map_location=torch.device("cpu"))


# ------------------------------------------------------------------------------
# Device fixture
# ------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def device():
    """Open a TT device once per session, enable persistent kernel cache, close on teardown."""
    dev = device_box.open({"device_id": 0, "l1_small_size": 29 * 1024}, enable_program_cache=True)
    # l1_small_size:
    # -------------A0: 39 * 1024;
    # -------------WH: 29 * 1024
    # ttnn.device.EnablePersistentKernelCache()
    yield dev
    device_box.close()


# ------------------------------------------------------------------------------
# Main test
# ------------------------------------------------------------------------------
def test_ssr_pcc(device):
    # --------------------- Build models --------------------- #
    cfg = Config.fromfile(COMMON_CONFIG_PATH)

    # Torch (reference)
    cfg.model.train_cfg = None
    ref_model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    # TT
    cfg.ttnn_model.pretrained = None
    cfg.data.test.test_mode = True
    tt_model = build_model(cfg.ttnn_model, test_cfg=cfg.get("test_cfg"))
    tt_model.img_backbone.load_config_dict(module_config)

    # --------------------- Prepare inputs ------------------- #
    # ref_img, tt_img, img_metas_np, img_metas_tt, ego_fut_cmd = make_inputs(1)
    dataloader = MockDataLoader(cfg, num_samples=NUM_SAMPLES)

    # --------------------- Transfer weights ----------------- #
    # Note: pass the same API structure the ModelProcessor expects.
    checkpoint = load_checkpoint(ref_model, PT_CHECKPOINT_PATH, map_location="cpu")
    tt_state_dict = torch.load(TT_CHECKPOINT_PATH, map_location="cpu")
    tt_model.load_state_dict(tt_state_dict, strict=False)

    embed_dict = load_embed_dict(EMBED_PATH)
    if embed_dict:
        tt_model.pts_bbox_head.transformer.convert_torch_embeds(**embed_dict)
        tt_model.pts_bbox_head.convert_torch_embeds(**embed_dict)

    # ----------------------- Compile ------------------------ #
    data_tt = extract_data_from_container(
        next(iter(dataloader)), tensor="tt", device=device, input_config=cfg.input_config
    )
    # First run to compile the model
    tt_out = tt_model(
        **data_tt,
        memory_config=memory_config,
        program_config=program_config,
    )
    for v in tt_out.values():
        if isinstance(v, ttnn.Tensor):
            ttnn.deallocate(v)

    # --------------------- Forward passes ------------------- #
    tt_model.eval()
    ref_model.eval()
    accumulated_time = 0.0
    for i, data in enumerate(dataloader):
        # Torch
        logger.info("Run SSR forward pass (Torch ref)")
        with torch.no_grad():
            data_pt = extract_data_from_container(deepcopy(data), tensor="pt")
            ref_out = ref_model(
                **data_pt,
            )
            for k, v in ref_out.items():
                assert torch.isfinite(v).all(), f"NaN/Inf in Torch output {k}"

        # TTNN
        logger.info("Run SSR forward pass (TTNN)")
        data_tt = extract_data_from_container(
            data, img_dram=data_tt["img"][0], tensor="tt", device=device, input_config=cfg.input_config
        )
        ttnn.synchronize_device(device)
        start_time = time.time()
        tt_out = tt_model(
            **data_tt,
            memory_config=memory_config,
            program_config=program_config,
        )
        ttnn.synchronize_device(device)
        elapsed_time = time.time() - start_time
        accumulated_time += elapsed_time

        # Clean up last tensors
        if i != NUM_SAMPLES - 1:
            for v in tt_out.values():
                if isinstance(v, ttnn.Tensor):
                    ttnn.deallocate(v)

        logger.info(f"Run {i+1}/{NUM_SAMPLES} took {elapsed_time:.4f} seconds")
    logger.info(f"Average TTNN time per run: {accumulated_time / NUM_SAMPLES:.4f} seconds")

    # --------------------- Assertions ----------------------- #
    # Compare ego future predictions between Torch and TT
    assert tt_out is not None, "TT output is None."
    for key in tt_out.keys():
        if key == "scene_query":
            continue
        _, pcc = compare_tensors(ref_out[key], tt_out[key], 0.97)
        assert pcc > 0.97, f"PCC below threshold: {pcc:.4f}; at key: {key}"
        logger.info(f"SSR test passed (PCC {pcc:.4f})")

    # Clean up last TT tensors after use
    for v in tt_out.values():
        if isinstance(v, ttnn.Tensor):
            ttnn.deallocate(v)


if __name__ == "__main__":
    dev = device_box.open({"device_id": 0, "l1_small_size": 29 * 1024}, enable_program_cache=True)
    test_ssr_pcc(device=dev)
    # l1_small_size:
    # -------------A0: 39 * 1024;
    # -------------WH: 29 * 1024
