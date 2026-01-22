import os
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
from test.SSR.tt import encoder as _tt_encoder
from test.SSR.tt import fpn as _tt_fpn
from test.SSR.tt import positional_encoding as _tt_posenc
from test.SSR.tt import resnet as _tt_resnet
from test.SSR.tt import spatial_cross_attention as _tt_spatial_cross_attention
from test.SSR.tt import temporal_self_attention as _tt_temporal_self_attention
from test.SSR.tt import tokenlearner as _tt_tokenlearner
from test.SSR.tt import transformer as _tt_transformer
from test.SSR.tt.runner.performant_runner import SSRPerformanceRunner
from test.SSR.tt.runner.performant_runner_infra import SSRPerformanceRunnerInfra
from test.SSR.tt.utils.misc import extract_data_from_container
from test.utils import tt2pt
from typing import List

import numpy as np
import pytest
import torch

# Project-specific imports
from bos_metal import compare_tensors, device_box, op
from loguru import logger
from mmcv import Config
from mmcv.cnn.bricks.registry import ATTENTION, NORM_LAYERS
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint
from mmdet.models.builder import BACKBONES, NECKS  # kept for side-effect registrations
from tqdm import tqdm
from tt.projects.configs import fpn, resnet50
from tt.projects.configs.ops_config import memory_config, program_config

import ttnn

# ------------------------------------------------------------------------------
# Global setup
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")

# If your repo needs a dynamic path, replace this with WORKING_DIR or remove it.
sys.path.append("")  # ⚠️ Prefer an absolute path or environment-based path

# Register custom ops
# NORM_LAYERS.register_module("LN_tt", module=op.LayerNorm)
# ATTENTION.register_module("MultiheadAttention_tt", module=op.MultiheadAttention)


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

NUM_SAMPLES = 10000
DOUBLE_CQ = True  # Whether to use double command queues for overlapping compute and data transfers
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


def load_embed_dict(path: str):
    if not path or not osp.exists(path):
        logger.warning("Embedding path missing; skipping embed conversion: %s", path)
        return None
    return torch.load(path, map_location=torch.device("cpu"))


class MockDataLoader:
    MAX_SAMPLE = 1_000
    def __init__(self, cfg, num_samples=2, seed=0):
        self.cfg = cfg
        self.num_samples = num_samples
        self._actual_num_samples = min(num_samples, self.MAX_SAMPLE)
        self.seed = seed
        self.dataset = [make_inputs(cfg, seed + i) for i in range(self._actual_num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx_ = idx % self.MAX_SAMPLE
        return {
            "img": [DataContainer(self.dataset[idx_][0])],
            "img_metas": [DataContainer(self.dataset[idx_][1])],
            "ego_fut_cmd": [DataContainer(self.dataset[idx_][2])],
        }

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= self._actual_num_samples:
            self._index = 0

        data = {  # list of DCs
            "img": [DataContainer(self.dataset[self._index][0])],
            "img_metas": [DataContainer(self.dataset[self._index][1])],
            "ego_fut_cmd": [DataContainer(self.dataset[self._index][2])],
        }

        self._index += 1

        return data


# ------------------------------------------------------------------------------
# Device fixture
# ------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def device():
    """Open a TT device once per session, enable persistent kernel cache, close on teardown."""
    device_dict = {"device_id": 0, "l1_small_size": 29 * 1024, "trace_region_size": 11685888}
    if DOUBLE_CQ:
        device_dict["num_command_queues"] = 2
    dev = device_box.open(device_dict, enable_program_cache=True)
    yield dev


# ------------------------------------------------------------------------------
# Main test
# ------------------------------------------------------------------------------
def test_ssr_pcc(device):
    # --------------------- Prepare inputs ------------------- #
    cfg = Config.fromfile(COMMON_CONFIG_PATH)
    dataloader = MockDataLoader(cfg, num_samples=NUM_SAMPLES)

    # --------------------- Build runner --------------------- #
    runner = SSRPerformanceRunner(
        device=device,
        common_config=cfg,
        backbone_config=resnet50.module_config,
        neck_config=fpn.module_config,
        embed_path=EMBED_PATH,
        pt_checkpoint_path=PT_CHECKPOINT_PATH,
        tt_checkpoint_path=TT_CHECKPOINT_PATH,
        memory_config=memory_config,
        program_config=program_config,
        double_cq=DOUBLE_CQ,
        dataset=dataloader.dataset,
    )

    # --------------------- Forward passes ------------------- #
    ## 1 - Run reference to get last input data output tensors
    logger.info("Run reference SSR-Net forward pass")
    data = dataloader[-1]
    runner(data, mode="reference")
    ref_out = runner.runner_infra.ref_out
    assert ref_out is not None, "Reference output is None."

    ## 2 - TTNN
    ## 2 - TTNN
    ### 2.1 - Compile
    logger.info("Compile TTNN SSR-Net model")
    data = next(iter(dataloader))
    runner(data, mode="compile")
    runner.dealloc_output()

    ### 2.2 - Warm up model
    logger.info("Warm up TTNN SSR-Net model")
    data = next(iter(dataloader))
    tt_out = runner(data, mode="normal", post_process=False, sample_idx=-1)
    assert tt_out is not None, "TT output is None."
    runner.dealloc_output()
    data = next(iter(dataloader))
    tt_out = runner(data, mode="normal", post_process=False, sample_idx=-1)
    assert tt_out is not None, "TT output is None."
    runner.dealloc_output()

    ### 2.3 - Trace capturing
    logger.info("Trace capture TTNN SSR-Net model")
    ttnn.synchronize_device(device)
    data = next(iter(dataloader))
    runner(data, mode="trace_capture")

    ### 2.3 - Run TT multiple times to get average time
    logger.info("Measure TTNN SSR-Net model performance")
    ttnn.synchronize_device(device)
    start_time = time.time()
    for i, data in enumerate(dataloader):
        logger.info(f"Running sample {i + 1}/{len(dataloader)}")
        tt_result, execution_time = runner(
            data, 
            mode="performant",
            sample_idx=i,
            pcc_threshold=0.98,
        )
    ttnn.synchronize_device(device)
    accumulated_time = time.time() - start_time
    logger.info(f"Average TTNN time per run: {accumulated_time / (NUM_SAMPLES):.4f} seconds")
