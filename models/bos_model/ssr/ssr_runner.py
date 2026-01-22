"""Thin API wrapper for running SSR TTNN inference from Python.

The goal of this module is to mirror the behavior of the shell helper
`scripts/run_ssr_demo.sh` while providing a programmatic interface that returns
useful runtime metrics (FPS) and optionally the raw TT output tensors.

Typical usage:

        from models.bos_model.ssr import ssr_runner

        result = ssr_runner(
                device_id=0,
                batch_size=1,
                num_iter=10,
                return_tt_results=False,
                validate=False,
                enable_persistent_kernel_cache=True,
        )

Environment requirements
------------------------
- ``WORKING_DIR`` must point to the project root that contains the ``tt`` and
    ``reference`` subdirectories as well as data checkpoints and embeddings.
    Example: WORKING_DIR=/home/bos/work/bhrc_user/nquang/tt-metal/models/bos_model/ssr/

Outputs
-------
The API returns a dictionary with:
- ``fps``: measured frames-per-second (float)
- ``output_tensors``: list of TT outputs when ``return_tt_results=True``;
    otherwise an empty list
"""

import importlib
import os
from copy import deepcopy

import mmcv
from bos_metal import device_box
from loguru import logger
from mmcv import Config
from tt.projects.configs import fpn, resnet50
from tt.projects.configs.ops_config import memory_config, program_config
from tt.projects.mmdet3d_plugin.SSR.runner import SSRPerformanceRunner

import ttnn
from models.bos_model.ssr.tt.pipeline import build_dataloader_from_cfg, register_ttnn_submodules, unset_env_vars

TT_CONFIG = f"{os.environ['WORKING_DIR']}/tt/projects/configs/SSR_e2e.py"
PT_CONFIG = f"{os.environ['WORKING_DIR']}/reference/projects/configs/SSR_e2e.py"
DATA_CONFIG = f"{os.environ['WORKING_DIR']}/tt/projects/configs/SSR_e2e.py"
EMBED_PATH = f"{os.environ['WORKING_DIR']}/data/embeddings/tensor_dict.pth"

# Mode-specific default checkpoints
DEFAULT_TT_CKPT = f"{os.environ['WORKING_DIR']}/data/ckpts/ssr_tt.pth"
DEFAULT_PT_CKPT = f"{os.environ['WORKING_DIR']}/data/ckpts/ssr_pt.pth"

TT_CHECKPOINT = f"{os.environ.get('TT_CHECKPOINT', DEFAULT_TT_CKPT)}"
PT_CHECKPOINT = f"{os.environ.get('PT_CHECKPOINT', DEFAULT_PT_CKPT)}"
DEFAULT_DEVICE_ID = 0
DEFAULT_L1_SMALL_SIZE = 29 * 1024  # in KB
DEFAULT_TRACE_REGION_SIZE = 11759616
DEFAULT_NUM_COMMAND_QUEUES = 2
DEFAULT_PCC_THRESHOLD = 0.98


def run_inference(runner, data_loader, *, num_iters, return_tt_results, repeat, **kwargs):
    """Run inference loop and optionally collect raw TT outputs.

    Parameters
    ----------
    runner : SSRPerformanceRunner
        Initialized runner object configured for TT inference.
    data_loader : torch.utils.data.DataLoader
        Dataloader that yields model-ready batches.
    num_iter : int
        Number of iterations to execute. This is capped at the dataset length
        for a single pass unless ``repeat=True``.
    return_tt_results : bool
        When True, accumulate and return TT outputs from each iteration.
    repeat : bool
        If True, keep looping over the dataloader until interrupted.
    **kwargs
        Extra arguments forwarded to the runner (e.g., ``debug``, ``visualize``,
        ``realtime``, ``use_bev``, ``validate``, ``pcc_threshold``).

    Returns
    -------
    tuple[list, list]
        A tuple of (output_tensors, execution_times).
    """

    tt_results, tt_times = [], []
    dataset = data_loader.dataset
    remaining = max(1, num_iters)

    try:
        while True:
            prog_bar = mmcv.ProgressBar(len(dataset))
            for i, data in enumerate(data_loader):
                tt_result, execution_time = runner(
                    data,
                    mode="performant",
                    sample_idx=i,
                    **kwargs,
                )

                if return_tt_results:
                    tt_results.extend(tt_result)

                tt_times.append(execution_time)
                prog_bar.update()

                remaining -= 1
                if remaining == 0:
                    break

            if remaining == 0 or not repeat:
                break

        if kwargs.get("visualize", False):
            runner.close_visualizer(**kwargs)
    except (KeyboardInterrupt, SystemExit):
        if kwargs.get("visualize", False):
            runner.close_visualizer(**kwargs)
        logger.info("KeyboardInterrupt or SystemExit detected, exiting...")

    return tt_results, tt_times


def load_model_config(kwargs):
    """Load and merge configuration files for torch and TTNN models.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments passed to ``ssr_runner``. Recognized keys:
        ``pt_config``, ``tt_config``, ``common_config``, ``cfg_options``.

    Returns
    -------
    mmcv.Config
        Combined config with ``pt_model`` and ``tt_model`` attached.
    """

    cfg = Config(dict())
    torch_cfg = Config.fromfile(kwargs.get("pt_config", PT_CONFIG))
    ttnn_cfg = Config.fromfile(kwargs.get("tt_config", TT_CONFIG))
    cfg.pt_model = torch_cfg
    cfg.tt_model = ttnn_cfg

    if kwargs.get("common_config", None) is not None:
        logger.info(f"Load common config from {kwargs.get('common_config')}")
        common_cfg = Config.fromfile(kwargs.get("common_config"))
        cfg.merge_from_dict(common_cfg)

    if kwargs.get("cfg_options", None) is not None:
        cfg.merge_from_dict(kwargs.get("cfg_options"))

    return cfg


def load_data_config(kwargs):
    """Load dataset configuration from file.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments passed to ``ssr_runner``. Recognized key:
        ``data_config`` (defaults to ``DATA_CONFIG``).

    Returns
    -------
    mmcv.Config
        Dataset configuration.
    """

    logger.info(f"Load data config from {kwargs.get('data_config', DATA_CONFIG)}")
    cfg = Config.fromfile(kwargs.get("data_config", DATA_CONFIG))
    return cfg


def ssr_runner(device_id, batch_size, num_iters, **kwargs):
    """Run SSR TTNN inference and return runtime metrics.

    Parameters
    ----------
    device_id : int
        Target TT device ID.
    batch_size : int
        Batch size for the dataloader.
    num_iter : int
        Number of inference iterations to execute. Use ``repeat=True`` to
        loop indefinitely until interrupted.
    kwargs : dict, optional
        Optional arguments mirroring the CLI/script options:

        - ``data_config``: path to data config (default: ``DATA_CONFIG``)
        - ``pt_config``: path to torch config (default: ``PT_CONFIG``)
        - ``tt_config``: path to TT config (default: ``TT_CONFIG``)
        - ``common_config``: shared overrides config
        - ``cfg_options``: dict merged into config
        - ``embed_path``: embeddings path (default: ``EMBED_PATH``)
        - ``pt_checkpoint``: torch checkpoint (default: env ``PT_CHECKPOINT`` or ``DEFAULT_PT_CKPT``)
        - ``tt_checkpoint``: TT checkpoint (default: env ``TT_CHECKPOINT`` or ``DEFAULT_TT_CKPT``)
        - ``l1_small_size``: L1 small size in KB (default: ``DEFAULT_L1_SMALL_SIZE``)
        - ``trace_region_size``: trace region size (default: ``DEFAULT_TRACE_REGION_SIZE``)
        - ``num_command_queues``: number of command queues (default: ``DEFAULT_NUM_COMMAND_QUEUES``)
        - ``enable_persistent_kernel_cache``: bool to enable persistent kernel cache
        - ``debug``: enable debug logging in configs
        - ``validate``: compare TT against torch reference
        - ``pcc_threshold``: PCC threshold used during validation
        - ``visualize``: enable visualization
        - ``realtime``: show inference directly instead of saving video
        - ``bev_map``: enable BEV map in visualization
        - ``repeat``: loop over the dataset indefinitely
        - ``return_tt_results``: collect TT output tensors to return

    Returns
    -------
    dict
        {"fps": float, "output_tensors": list}
    """

    unset_env_vars()
    register_ttnn_submodules()
    importlib.reload(resnet50)
    importlib.reload(fpn)

    logger.info(f"Using device: {device_id}")
    device = device_box.open(
        {
            "device_id": device_id,
            "l1_small_size": kwargs.get("l1_small_size", DEFAULT_L1_SMALL_SIZE),
            "trace_region_size": kwargs.get("trace_region_size", DEFAULT_TRACE_REGION_SIZE),
            "num_command_queues": kwargs.get("num_command_queues", DEFAULT_NUM_COMMAND_QUEUES),
        },
        enable_program_cache=True,
    )

    if kwargs.get("enable_persistent_kernel_cache", False):
        logger.info("Enable persistent kernel cache")
        ttnn.device.EnablePersistentKernelCache()

    logger.info("Build dataset and dataloader")
    data_config = load_data_config(kwargs)
    _, data_loader = build_dataloader_from_cfg(data_config, batch_size)

    logger.info("Building models")
    cfg = load_model_config(kwargs)
    cfg.tt_model.debug = kwargs.get("debug", False)

    runner = SSRPerformanceRunner(
        device=device,
        common_config=cfg,
        backbone_config=resnet50.module_config,
        neck_config=fpn.module_config,
        embed_path=kwargs.get("embed_path", EMBED_PATH),
        pt_checkpoint_path=kwargs.get("pt_checkpoint", PT_CHECKPOINT),
        tt_checkpoint_path=kwargs.get("tt_checkpoint", TT_CHECKPOINT),
        memory_config=memory_config,
        program_config=program_config,
        double_cq=True,
        dataset=data_loader.dataset,
    )

    logger.info("Compile TTNN SSR-Net model")
    data = next(iter(data_loader))
    runner(data, mode="compile")
    runner.dealloc_output()

    logger.info("Warm up TTNN SSR-Net model")
    for _ in range(2):
        data = next(iter(data_loader))
        tt_out = runner(data, mode="normal", post_process=False, sample_idx=-1)
        assert tt_out is not None, "TT output is None."
        runner.dealloc_output()

    logger.info("Trace capture TTNN SSR-Net model")
    ttnn.synchronize_device(device)
    data = next(iter(data_loader))
    runner(data, mode="trace_capture")

    logger.info("Start inference on TTNN SSR-Net model")
    ttnn.synchronize_device(device)
    outputs = run_inference(
        runner,
        data_loader,
        num_iters=num_iters,
        return_tt_results=kwargs.get("return_tt_results", False),
        repeat=kwargs.get("repeat", False),
        debug=kwargs.get("debug", False),
        visualize=kwargs.get("visualize", False),
        realtime=kwargs.get("realtime", False),
        use_bev=kwargs.get("bev_map", False),
        validate=kwargs.get("validate", False),
        pcc_threshold=kwargs.get("pcc_threshold", DEFAULT_PCC_THRESHOLD),
    )

    runner.release()
    device_box.close()

    warmup_skip = 5
    fps = (
        (num_iters - warmup_skip) / sum(outputs[1][warmup_skip:])
        if num_iters > warmup_skip
        else num_iters / sum(outputs[1])
    )

    return {
        "fps": fps,
        "output_tensors": outputs[0],
    }
