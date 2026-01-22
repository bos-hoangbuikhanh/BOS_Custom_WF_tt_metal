# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PyTest for the TT-NN ResNet-50 backbone with Metal Trace.

*Importance note: Before running the script, device synchronization statements must be commented in resnet.py

• Builds the standard MMDetection ResNet-50 and its TT-NN counterpart.
• Copies the processed state-dict from PyTorch → TT-NN.
• Feeds a random batch through both implementations.
• Compares output feature-maps (PCC > 0.98 and shape equality).
"""

import os
import os.path as osp
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from test.builder import build_backbone
from test.SSR.tt import resnet as _tt_resnet
from test.SSR.tt.utils.misc import extract_data_from_container
from typing import List

import numpy as np
import pytest
import torch

# Project-specific imports
from bos_metal import compare_tensors, device_box, op
from loguru import logger
from mmcv import Config
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint
from tt.projects.configs.resnet50 import module_config

import ttnn

# Force import to register modules (side effects for registries)


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

NUM_SAMPLES = 5
# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Build “safe” metadata (avoid geometric NaNs)
# ------------------------------------------------------------------------------
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
    yield dev
    device_box.close()


# ------------------------------------------------------------------------------
# Main test
# ------------------------------------------------------------------------------
class PTResNet50(torch.nn.Module):
    def __init__(self, img_backbone):
        super().__init__()
        self.img_backbone = img_backbone

    def forward(self, img):
        return self.img_backbone(img)


class TTResNet50(op.BaseModule):
    def __init__(self, img_backbone):
        super().__init__()
        self.img_backbone = img_backbone
        # self.img_backbone.eval()

    def forward(self, img):
        return self.img_backbone(img)


def test_resnet_backbone_pcc(device):
    # --------------------- Build models --------------------- #
    cfg = Config.fromfile(COMMON_CONFIG_PATH)

    # Torch (reference)
    ref_model = build_backbone(cfg.model.img_backbone)
    ref_model = PTResNet50(ref_model)

    # TT
    cfg.ttnn_model.pretrained = None
    tt_resnet_model = build_backbone(cfg.ttnn_model.img_backbone)
    tt_resnet_model.load_config_dict(module_config)
    tt_model = TTResNet50(tt_resnet_model)

    # --------------------- Prepare inputs ------------------- #
    cfg.data.test.test_mode = True
    dataloader = MockDataLoader(cfg, num_samples=NUM_SAMPLES)

    # --------------------- Transfer weights ----------------- #
    # Note: pass the same API structure the ModelProcessor expects.
    load_checkpoint(ref_model, PT_CHECKPOINT_PATH, map_location="cpu")
    tt_state_dict = torch.load(TT_CHECKPOINT_PATH, map_location="cpu")
    tt_model.load_state_dict(tt_state_dict, strict=False)

    # -------------------- Compile and capture trace ------------------------ #
    data_tt = extract_data_from_container(
        next(iter(dataloader)), tensor="tt", device=device, input_config=cfg.input_config
    )
    # First run to compile the model
    ttnn_output_tensor = tt_model(data_tt["img"][0])[-1]
    ttnn.deallocate(ttnn_output_tensor)

    # Capture the trace of the model
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    logger.info("Capturing trace for ResNet-50 backbone...")
    tt_out = tt_model(data_tt["img"][0])[-1]
    ttnn.end_trace_capture(device, tid, cq_id=0)

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
                data_pt["img"][0][0],
            )[-1]

        # TTNN
        logger.info("Run SSR forward pass (TTNN)")
        data_tt = extract_data_from_container(
            data, img_dram=data_tt["img"][0], tensor="tt", device=device, input_config=cfg.input_config
        )
        ttnn.synchronize_device(device)
        start_time = time.time()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        elapsed_time = time.time() - start_time
        accumulated_time += elapsed_time
        logger.info(f"Run {i+1}/{NUM_SAMPLES} took {elapsed_time:.4f} seconds")
    logger.info(f"Average TTNN time per run: {accumulated_time / NUM_SAMPLES:.4f} seconds")

    # --------------------- Assertions ------------------------------------- #
    # Convert TT output back to PyTorch NCHW
    tt_out = ttnn.to_torch(tt_out)
    tt_out = tt_out.reshape(ref_out.shape[0], ref_out.shape[2], ref_out.shape[3], ref_out.shape[1])
    tt_out = tt_out.permute(0, 3, 1, 2).contiguous()
    assert ref_out.shape == tt_out.shape, "Feature-map shape mismatch"
    _, pcc = compare_tensors(ref_out, tt_out)
    assert pcc > 0.98, f"PCC below threshold: {pcc:.4f}"

    logger.info(f"ResNet backbone test passed (PCC {pcc:.4f})")
