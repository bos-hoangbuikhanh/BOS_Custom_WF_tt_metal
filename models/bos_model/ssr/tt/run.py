# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Nhat Nguyen
# ---------------------------------------------

import argparse
from loguru import logger 
import os
import os.path as osp
import sys
import time
import warnings
import pathlib

sys.path.append("")

import ttnn
import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info

from bos_metal import device_box, op
from pipeline import (
    build_dataloader_from_cfg,
    unset_env_vars,
    register_ttnn_submodules
)
from tt.projects.configs import fpn, resnet50
from tt.projects.configs.ops_config import memory_config, program_config
from reference.projects.mmdet3d_plugin.datasets import VADCustomNuScenesDataset
from tt.projects.mmdet3d_plugin.SSR.runner import SSRPerformanceRunner

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1])) 

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore")

# Constants
DEFAULT_DEVICE_ID = 0
DEFAULT_L1_SMALL_SIZE = 29 * 1024  # in KB
DEFAULT_TRACE_REGION_SIZE = 11759616
DEFAULT_NUM_COMMAND_QUEUES = 2
DEFAULT_PCC_THRESHOLD = 0.98

def get_output_path(out_dir='generated/video'):
    current_path = os.getcwd()
    current_path = current_path.rsplit('/', 1)[0] + '/'
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(current_path, out_dir)

    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(out_dir, f"realtime_{ts}.mp4")
    return video_path

def run_inference(runner, data_loader, **kwargs):
    tt_results, tt_times = [], []
    dataset = data_loader.dataset

    try:
        while True:
            prog_bar = mmcv.ProgressBar(len(dataset))
            for i, data in enumerate(data_loader):
                tt_result, execution_time = runner(data, mode="performant", sample_idx=i, **kwargs)

                if not kwargs.get("repeat", False): 
                    tt_results.extend(tt_result)

                tt_times.append(execution_time)
                prog_bar.update()

            if not kwargs.get("repeat", False): 
                break

        if kwargs.get("visualize", False):
            runner.close_visualizer(**kwargs)
    except (KeyboardInterrupt, SystemExit):
        if kwargs.get("visualize", False):
            runner.close_visualizer(**kwargs)
        logger.info("KeyboardInterrupt or SystemExit detected, exiting...")

    return tt_results


def parse_and_validate_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    # Configs
    parser.add_argument("--data_config", help="Data config file path")
    parser.add_argument("--pt_config", help="Torch model config file", default=None)
    parser.add_argument("--tt_config", help="TTNN model config file", default=None)
    parser.add_argument("--common_config", help="shared config file", default=None)
    # Checkpoints
    parser.add_argument("--tt_checkpoint", help="TTNN checkpoint file")
    parser.add_argument("--pt_checkpoint", help="Torch checkpoint file", default=None)
    parser.add_argument("--embeddings", help="embeddings file")
    # Device
    parser.add_argument("--device_id", type=int, default=DEFAULT_DEVICE_ID, help="Device ID used for inference")
    parser.add_argument("--l1_small_size", type=int, default=DEFAULT_L1_SMALL_SIZE, help="L1 small size")
    parser.add_argument("--trace_region_size", type=int, default=DEFAULT_TRACE_REGION_SIZE, help="Trace region size")
    parser.add_argument("--num_command_queues", type=int, default=DEFAULT_NUM_COMMAND_QUEUES, help="Number of command queues")
    parser.add_argument("--enable_persistent_kernel_cache", action="store_true", help="Enable persistent kernel cache")
     # Validation
    parser.add_argument("--validate", action="store_true", help="Whether to validate the ttnn model with the torch model")
    parser.add_argument("--pcc_threshold", type=float, default=DEFAULT_PCC_THRESHOLD, help="PCC threshold for validation")
    # Visualize
    parser.add_argument("--visualize", action="store_true",
                    help="enable realtime visualization")
    parser.add_argument("--bev_map", action="store_true",
                        help="enable BEV map during realtime visualization", default=False)
    parser.add_argument("--realtime", action="store_true",
                        help="show inference directly or save it as video")
    # Others
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--return_model", type=int, default=0)
    parser.add_argument("--repeat", action="store_true",
                        help="repeat demo until user interrupts")

    args = parser.parse_args()

    # Mandatory arguments
    required_args = {
        "data_config": args.data_config,
        "device_id": args.device_id,
        "l1_small_size": args.l1_small_size,
        "trace_region_size": args.trace_region_size,
        "num_command_queues": args.num_command_queues,
    }
    for name, value in required_args.items():
        assert value is not None, f"Please specify --{name}"

    # Model requirements
    assert args.tt_config or args.pt_config, "Please specify --tt_config or --pt_config"
    if args.validate:
        assert (args.tt_config is not None and args.pt_config is not None), \
            "To validate, both --tt_config and --pt_config must be specified."
    
    # Visualization 
    if args.visualize:
        assert (args.tt_config is not None), \
            "To visualize, --tt_config must be specified."
            
    # Checkpoint requirements
    if args.tt_config is not None: 
        assert args.tt_checkpoint is not None, "Please specify --tt_checkpoint"
        assert args.embeddings is not None, "Please specify --embeddings"
    if args.pt_config is not None: 
        assert args.pt_checkpoint is not None, "Please specify --pt_checkpoint"

    # Others
    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    
    if args.seed:
        torch.manual_seed(args.seed)

    return args


def load_model_config(args):
    """Load and merge configuration files for torch and ttnn models."""
    cfg = Config(dict())
    if args.pt_config:
        torch_cfg = Config.fromfile(args.pt_config)
        cfg.pt_model = torch_cfg
    if args.tt_config:
        ttnn_cfg = Config.fromfile(args.tt_config)
        cfg.tt_model = ttnn_cfg
        ttnn_cfg = Config.fromfile(args.tt_config)
    
    # Common and custom options
    if args.common_config is not None:
        logger.info(f"Load common config from {args.common_config}")
        common_cfg = Config.fromfile(args.common_config)
        cfg.merge_from_dict(common_cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # Handle custom imports
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
        logger.info(f"Import modules from string list: {cfg['custom_imports']['imports'][0]}")

    if hasattr(cfg, "plugin_dir"):
        # import modules from plguin/xx, registry will be updated
        import importlib
        plugin_dir = cfg.plugin_dir
        if isinstance(plugin_dir, str):
            plugin_dir = [plugin_dir]
        for plugin in plugin_dir:
            _module_dir = os.path.dirname(plugin)
            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            logger.info(f"Import modules from {plugin}, _module_path: {_module_path}")
            _ = importlib.import_module(_module_path)
            
    return cfg


def load_data_config(args):
    """Load and merge configuration files."""
    if args.data_config is None:
        return Config()
    
    # Load and merge configurations
    logger.info(f"Load data config from {args.data_config}")
    cfg = Config.fromfile(args.data_config)

    return cfg


def main():
    unset_env_vars()
    register_ttnn_submodules()
    args = parse_and_validate_args()

    # Device
    logger.info(f"Using device: {args.device_id}")
    device = device_box.open({
        "device_id": args.device_id,
        "l1_small_size": args.l1_small_size,
        "trace_region_size": args.trace_region_size,
        "num_command_queues": args.num_command_queues,
    }, enable_program_cache=True)
    # if args.enable_persistent_kernel_cache:
    #     ttnn.device.EnablePersistentKernelCache()

    # Build dataloader
    logger.info("Build dataset and dataloader")
    data_config = load_data_config(args)
    _, data_loader = build_dataloader_from_cfg(data_config, 1)

    # build the model and load checkpoint
    logger.info("Building models")
    cfg = load_model_config(args)
    cfg.tt_model.debug = args.debug

    runner = SSRPerformanceRunner(
        device=device,
        common_config=cfg,
        backbone_config=resnet50.module_config,
        neck_config=fpn.module_config,
        embed_path=args.embeddings,
        pt_checkpoint_path=args.pt_checkpoint,
        tt_checkpoint_path=args.tt_checkpoint,
        memory_config=memory_config,
        program_config=program_config,
        double_cq=True,
        dataset=data_loader.dataset,
    )

    if args.return_model:
        return runner, data_loader, cfg

    ### Compile model
    logger.info("Compile TTNN SSR-Net model")
    data = next(iter(data_loader))
    runner(data, mode="compile")
    runner.dealloc_output()

    ### Warm up model
    logger.info("Warm up TTNN SSR-Net model")
    data = next(iter(data_loader))
    tt_out = runner(data, mode="normal", post_process=False)
    assert tt_out is not None, "TT output is None."
    runner.dealloc_output()

    # ### 2.3 - Trace capturing
    logger.info("Trace capture TTNN SSR-Net model")
    ttnn.synchronize_device(device)
    data = next(iter(data_loader))
    runner(data, mode="trace_capture")

    ### 2.4 - Inference
    logger.info("Start inference on TTNN SSR-Net model")
    ttnn.synchronize_device(device)
    outputs = run_inference(
        runner, 
        data_loader,
        debug=args.debug, 
        visualize=args.visualize, 
        realtime=args.realtime,
        use_bev=args.bev_map,
        validate=args.validate,
        pcc_threshold=args.pcc_threshold,
        repeat=args.repeat,
    )

    # Save results
    tmp = {}
    tmp["bbox_results"] = outputs
    outputs = tmp
    rank, _ = get_dist_info()
    if rank == 0 and args.out:
        logger.info(f"\nwriting results to {args.out}")
        # assert False
        if isinstance(outputs, list):
            mmcv.dump(outputs, args.out)
        else:
            mmcv.dump(outputs["bbox_results"], args.out)


if __name__ == "__main__":
    main()
