# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Nhat Nguyen
# ---------------------------------------------

import sys
import argparse
import os
import logging
import gc
import warnings
from pathlib import Path
from copy import deepcopy
sys.path.append("")

import ttnn
import torch
import mmcv
from mmcv import Config, DictAction

from bos_metal import op
from pipeline import (
    unset_env_vars,
)
from tt.projects.configs.ops_config import memory_config, program_config
from tt.projects.mmdet3d_plugin.SSR.utils.misc import (
    extract_data_from_container,
)


torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore")

# Logger
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
logging.getLogger("shapely.geos").setLevel(logging.WARNING)

# Get base dir
BASE_DIR = os.environ.get("WORKING_DIR", None)
if BASE_DIR is None:
    raise RuntimeError("WORKING_DIR SHOULD NOT BE NONE")
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
        
DATA_ROOT = os.environ.get("DATA_ROOT", f"{BASE_DIR}/data/dataset/nuscenes")
if not os.path.exists(DATA_ROOT):
    raise RuntimeError(f"Data root {DATA_ROOT} does not exist")
 

# ------------------------------- Import roots
def _ensure_import_roots(prefer, remove) -> None:
    """Keep only one project tree active on sys.path to avoid registry clashes."""
    sp, sr = str(prefer), str(remove)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    if sr in sys.path:
        sys.path.remove(sr)


# ------------------------------- Registry hygiene
def purge_projects_modules() -> None:
    """Remove imported project modules so re-import loads another tree."""
    for k in list(sys.modules.keys()):
        if k == "projects" or k.startswith("projects."):
            del sys.modules[k]


def nuke_registry_entries_and_scope(
    prefix: str = "projects.mmdet3d_plugin", scope_name: str = "projects"
):
    """Remove registered classes under `prefix` and drop child scope `projects`."""
    nuked_classes, nuked_scopes = 0, 0
    try:
        from mmcv.utils import Registry as MMCVRegistry
    except Exception:
        MMCVRegistry = None
    try:
        from mmengine.registry import Registry as MMEngineRegistry
    except Exception:
        MMEngineRegistry = None

    for obj in gc.get_objects():
        try:
            is_reg = (
                (MMCVRegistry and isinstance(obj, MMCVRegistry))
                or (MMEngineRegistry and isinstance(obj, MMEngineRegistry))
            )
            if not is_reg:
                continue

            module_dict = getattr(obj, "module_dict", None)
            if isinstance(module_dict, dict):
                for name, cls in list(module_dict.items()):
                    modname = getattr(cls, "__module__", "")
                    if isinstance(modname, str) and modname.startswith(prefix):
                        module_dict.pop(name, None)
                        nuked_classes += 1

            children = getattr(obj, "children", None)
            if isinstance(children, dict) and scope_name in children:
                del children[scope_name]
                nuked_scopes += 1
        except Exception:
            pass
    return nuked_classes, nuked_scopes

def make_paths(base_dir):
    from dataclasses import dataclass
    base = Path(base_dir).resolve()
    
    @dataclass(frozen=True)
    class Paths:
        base: Path
        ref_root: Path
        tt_root: Path
        ckpt_pt: Path
        ckpt_tt: Path
        embeddings: Path
        
    return Paths(
        base=base,
        ref_root=base / "reference",
        tt_root=base / "tt",
        ckpt_pt=base / "data" / "ckpts" / "ssr_pt.pth",
        ckpt_tt=base / "data" / "ckpts" / "ssr_tt.pth",
        embeddings=base / "data" / "embeddings" / "tensor_dict.pth",
    )

# ------------------------------- Model loaders
def _import_ssr_modules() -> None:
    """Force-register SSR modules in case config lacks custom_imports."""
    import projects.mmdet3d_plugin.SSR.SSR  # noqa: F401
    import projects.mmdet3d_plugin.SSR.SSR_transformer  # noqa: F401
    import projects.mmdet3d_plugin.SSR.modules.encoder  # noqa: F401
    import projects.mmdet3d_plugin.SSR.modules.spatial_cross_attention  # noqa: F401
    import projects.mmdet3d_plugin.SSR.modules.temporal_self_attention  # noqa: F401

def _build_cfg_options(data_root):
    data_root = str(data_root)
    ann_val = f"{data_root}/vad_nuscenes_infos_temporal_val.pkl"
    map_ann_val = f"{data_root}/nuscenes_map_anns_val.json"
    return [
        f"data.test.data_root={data_root}",
        f"data.test.ann_file={ann_val}",
        f"data.test.map_ann_file={map_ann_val}",
        f"data.data_root={data_root}",
    ]
    
def load_torch_model(data_root, paths):
    """Return (torch_model, dataset) by invoking reference/run.py main()."""

    _ensure_import_roots(prefer=paths.ref_root, remove=paths.tt_root)
    purge_projects_modules()
    nuke_registry_entries_and_scope()
    _import_ssr_modules()
    from reference.run import main as TORCH_MAIN

    sys.argv = [
        "run.py",
        "--config",
        str(paths.ref_root / "projects" / "configs" / "SSR_e2e.py"),
        "--checkpoint",
        str(paths.ckpt_pt),
        "--launcher",
        "none",
        "--eval",
        "bbox",
        "--return_model",
        "1",
        "--cfg-options",
        *_build_cfg_options(data_root),
    ]
    return TORCH_MAIN()


def load_ttnn_model(data_root, paths):
    """Return (ttnn_model, data_loader) by invoking tt/run.py main()."""
    _ensure_import_roots(prefer=paths.tt_root, remove=paths.ref_root)
    purge_projects_modules()
    nuke_registry_entries_and_scope()
    _import_ssr_modules()
    from tt.run import main as TTNN_MAIN
    from tt.run import device

    sys.argv = [
        "run.py",
        "--config",
        str(paths.tt_root / "projects" / "configs" / "SSR_e2e.py"),
        "--checkpoint",
        str(paths.ckpt_tt),
        "--embeddings",
        str(paths.embeddings),
        "--launcher",
        "none",
        "--eval",
        "bbox",
        "--return_model",
        "1",
        "--cfg-options",
        *_build_cfg_options(data_root),
    ]
    
    tt_model, tt_dataloader, cfg = TTNN_MAIN()
    return tt_model, tt_dataloader, cfg, device


def run_test(torch_model, ttnn_model, cfg, patch_data, device, pcc_threshold=0.98, data_loader=None):

    logging.disable(logging.CRITICAL)

    tt_results, torch_results = [], []
    # torch_batch = patch_data
    # tt_batch = deepcopy(patch_data)
    torch_model.eval()

    torch_model.eval()
    try:
        if data_loader is not None:
            for i, torch_batch in enumerate(data_loader):
                tt_batch = deepcopy(torch_batch)
                logger.info("Running torch forward")
                torch_data = extract_data_from_container(torch_batch, tensor='pt')
                with torch.no_grad():
                    torch_result = torch_model(
                        return_loss=False, 
                        rescale=True, 
                        **torch_data
                    )
                torch_results.extend(torch_result)

                logger.info("Running ttnn forward")
                tt_data = extract_data_from_container(tt_batch, tensor='tt', device=device,
                                                    input_config=cfg.input_config)
                tt_result = ttnn_model(
                    rescale=True,
                    **tt_data,
                    memory_config=memory_config,
                    program_config=program_config,
                )
                tt_results.extend(tt_result)
                
                tt_fut_preds = torch_result[0]['pts_bbox']['ego_fut_preds']
                pt_fut_preds = tt_result[0]['pts_bbox']['ego_fut_preds']
                print(f"Patch data {i}")
                passed, msg = op.compare_tensors(pt_fut_preds, tt_fut_preds, pcc=pcc_threshold)

    except (KeyboardInterrupt, SystemExit):
        print("KeyboardInterrupt or SystemExit detected, exiting...")

    return tt_results, torch_results


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("--config", help="version test config file path")
    parser.add_argument("--common_config", help="shared config file", default=None)
    parser.add_argument("--torch_config", help="Torch model config file", default=None)
    parser.add_argument("--ttnn_config", help="TTNN model config file", default=None)
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument("--embeddings", help="embeddings file")
    parser.add_argument("--patch", help="patch file")
    parser.add_argument(
        "--pcc_threshold", type=float, default=0.98,
        help="PCC threshold for tensor comparison"
    )
    parser.add_argument(
        "--json_dir", help="json parent dir name file"
    )  # NOTE: json file parent folder name
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
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
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--return_model", type=int, default=0)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options

    return args


def main():
    unset_env_vars()
    args = parse_args()
    paths = make_paths(BASE_DIR)
    data_root = Path(DATA_ROOT)

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")
        
    assert os.path.exists(args.patch), f"Patch file {args.patch} does not exist."
    if args.patch is not None and not args.patch.endswith((".pt")):
        raise ValueError("The patch file must be a pt file.")
    patch_data = torch.load(args.patch)
    pcc_threshold = args.pcc_threshold
    
    # Load models & data (pass `paths` in)
    torch_model, dataset = load_torch_model(data_root, paths)
    ttnn_model, data_loader, cfg, device = load_ttnn_model(data_root, paths)

    # Inference
    logger.info("Running functional test...")
    ttnn_outputs, torch_outputs = run_test(
        torch_model,
        ttnn_model,
        cfg,
        patch_data,
        device=device,
        pcc_threshold=pcc_threshold,
        data_loader=data_loader
    )


if __name__ == "__main__":
    main()
