# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import sys
sys.path.append('')
import argparse
import mmcv
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import (get_dist_info, load_checkpoint, wrap_fp16_model)

from reference.projects.mmdet3d_plugin.datasets.builder import build_dataloader
from reference.projects.mmdet3d_plugin.SSR.utils.builder import build_model
from reference.projects.mmdet3d_plugin.datasets import VADCustomNuScenesDataset
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

import warnings
warnings.filterwarnings("ignore")


import platform
from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))
    
    
def extract_data_from_container(data):
    """Extract data from DataContainer."""
    data["img_metas"] = data["img_metas"][0].data
    # data["points"] = data["points"][0].data
    data["gt_bboxes_3d"] = data["gt_bboxes_3d"][0].data
    data["gt_labels_3d"] = data["gt_labels_3d"][0].data
    data["img"] = data["img"][0].data
    # data["fut_valid_flag"] = data["fut_valid_flag"][0].data
    data["ego_his_trajs"] = data["ego_his_trajs"][0].data
    data["ego_fut_trajs"] = data["ego_fut_trajs"][0].data
    # data["ego_fut_masks"] = data["ego_fut_masks"][0].data
    data["ego_fut_cmd"] = data["ego_fut_cmd"][0].data
    data["ego_lcf_feat"] = data["ego_lcf_feat"][0].data
    data["gt_attr_labels"] = data["gt_attr_labels"][0].data
    # data["gt_attr_labels"] = data["gt_attr_labels"][0]
    data["map_gt_labels_3d"] = data["map_gt_labels_3d"].data[0]
    data["map_gt_bboxes_3d"] = data["map_gt_bboxes_3d"].data[0]
    return data


OBJECTSAMPLERS = Registry('Object sampler')
 
def build_dataset(cfg, default_args=None):
    return build_from_cfg(cfg, DATASETS, default_args)


def single_gpu_test(model,
                    data_loader):
    
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    try:
        for i, data in enumerate(data_loader):
            data = extract_data_from_container(data)
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            results.extend(result)

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
    except KeyboardInterrupt:
        print('Keyboard interrupt, exiting...')
        
    # [(100, 100), ...] per camera frequency counts captured during encoder forward
    heatmaps_list = model.pts_bbox_head.transformer.encoder.layers[0].attentions[1]._heatmaps_list

    if heatmaps_list:
        from pathlib import Path
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt

        output_dir = Path('logs/run_heatmaps')
        output_dir.mkdir(parents=True, exist_ok=True)

        final_heatmap = None
        for cam_idx, heatmap in enumerate(heatmaps_list):
            if isinstance(heatmap, torch.Tensor):
                heatmap_np = heatmap.detach().cpu().float().numpy()
            else:
                heatmap_np = np.asarray(heatmap, dtype=np.float32)

            heatmap_np = np.clip(heatmap_np, a_min=0.0, a_max=None).astype(
                np.int32, copy=False)
            if heatmap_np.ndim != 2 or heatmap_np.shape != (100, 100):
                continue
            if final_heatmap is None:
                final_heatmap = heatmap_np.copy()
            else:
                final_heatmap += heatmap_np

            save_path = output_dir / f'camera_{cam_idx}_heatmap.png'
            fig, ax = plt.subplots(figsize=(11, 11))
            sns.heatmap(
                heatmap_np,
                annot=True,
                # fmt='.2f',
                annot_kws={'size': 3},
                square=True,
                cbar=True,
                ax=ax,
            )
            fig.tight_layout()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            # save_heatmap_png(heatmap_np, save_path)
        if final_heatmap is not None:
            final_path = output_dir / 'final_heatmap.png'
            fig, ax = plt.subplots(figsize=(11, 11))
            sns.heatmap(
                final_heatmap,
                annot=True,
                annot_kws={'size': 3},
                square=True,
                cbar=True,
                ax=ax,
            )
            fig.tight_layout()
            fig.savefig(final_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--json_dir', help='json parent dir name file') # NOTE: json file parent folder name
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--return_model', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    # model = MMDataParallel(model, device_ids=[0])
    if args.return_model:
        return model, dataset
    outputs = single_gpu_test(model, data_loader)

    tmp = {}
    tmp['bbox_results'] = outputs
    outputs = tmp
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            # assert False
            if isinstance(outputs, list):
                mmcv.dump(outputs, args.out)
            else:
                mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs['bbox_results'], **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs['bbox_results'], **eval_kwargs))
    

if __name__ == '__main__':
    main()
    
    
