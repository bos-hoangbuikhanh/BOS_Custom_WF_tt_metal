import logging
import os
from tt.projects.mmdet3d_plugin.SSR.utils.builder import build_model

import torch
from bos_metal import compare_tensors, op, ttnn
from mmcv.cnn.bricks.registry import ATTENTION, NORM_LAYERS
from mmcv.runner import load_checkpoint

from contextlib import contextmanager
import threading

class DoubleStorage:
    def __init__(self):
        self.data = [None, None]
        self.read_idx = 0
        self.write_idx = 0

    def get(self, mode="read"):
        if mode == "read":
            assert self.data[self.read_idx] is not None, "Data not set before get."
            output = self.data[self.read_idx]
            self.read_idx = 1 - self.read_idx
        elif mode == "write":
            # assert self.data[self.write_idx] is not None, "Data not set before get."
            output = self.data[self.write_idx]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return output

    def set(self, data):
        self.data[self.write_idx] = data
        return_idx = self.write_idx
        self.write_idx = 1 - self.write_idx
        return return_idx

class InvocationBuffer:
    def __init__(self, device, target_cq_id: int):
        self.device = device
        self.target_cq_id = target_cq_id
        self._blocks = []
        self._tls = threading.local()

    @contextmanager
    def add_block(self, priority=0):
        # fence_id just needs to be monotonic for events recorded on same CQ
        ops = []
        self._tls.ops = ops

        # the function to enqueue work lazily
        def defer(fn):
            # fn must be a zero-arg callable doing the actual work when invoked later
            ops.append(fn)

        try:
            yield defer
        finally:
            self._blocks.append((priority, ops))
            self._tls.ops = None

    def flush(self, num_blocks=99):
        def key(b):
            return b[0]

        self._blocks.sort(key=key)
        for i, (priority, ops) in enumerate(self._blocks):
            # gate the block behind the event on CQ target
            if i >= num_blocks:
                break
            for op in ops:
                op()
        for _ in range(min(num_blocks, len(self._blocks))):
            self._blocks.pop(0)

class SSRPerformanceRunnerInfra:
    def __init__(
        self,
        device,
        common_config=None,
        backbone_config=None,
        neck_config=None,
        embed_path=None,
        pt_checkpoint_path=None,
        tt_checkpoint_path=None,
        memory_config=None,
        program_config=None,
    ):
        # 1 - Initialize object attributes
        self.memory_config = memory_config
        self.program_config = program_config

        assert common_config is not None, "common_config is None."
        assert backbone_config is not None, "backbone_config is None."
        assert neck_config is not None, "neck_config is None."
        assert tt_checkpoint_path is not None, "tt_checkpoint_path is None."
        if pt_checkpoint_path is not None: # Only check path existance if Pytorch model is used as a reference model
            assert os.path.exists(pt_checkpoint_path), f"PT checkpoint path does not exist: {pt_checkpoint_path}"
        assert os.path.exists(tt_checkpoint_path), f"TT checkpoint path does not exist: {tt_checkpoint_path}"
        assert os.path.exists(embed_path), f"Embed path does not exist: {embed_path}"
        assert memory_config is not None, "memory_config is None."
        assert program_config is not None, "program_config is None."

        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.num_devices = device.get_num_devices()

        # 2 - Initialize logger
        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("shapely.geos").setLevel(logging.WARNING)

        # 3 - Initialize output tensors
        self.ref_out = None
        self.tt_out = None

        # 4 - Initialize models
        ## 4.1 - Torch (reference)
        # common_config.model.train_cfg = None # Assigned in caller
        self.ref_model = None
        if common_config.get("pt_model", None) is not None:
            self.ref_model = build_model(common_config.pt_model.model, test_cfg=common_config.get("test_cfg"))
            self.ref_model.eval()

        ## 4.2 - TT
        self.tt_model = build_model(common_config.tt_model.model, test_cfg=common_config.get("test_cfg"))
        self.tt_model.img_backbone.load_config_dict(backbone_config)
        self.tt_model.img_neck.load_config_dict(neck_config)
        self.tt_model.eval()

        # 5 - Load parameters
        if pt_checkpoint_path is not None:
            load_checkpoint(self.ref_model, pt_checkpoint_path, map_location="cpu")
        tt_state_dict = torch.load(tt_checkpoint_path, map_location="cpu")
        self.tt_model.load_state_dict(tt_state_dict, strict=False)

        embed_dict = torch.load(embed_path, map_location=torch.device("cpu")) if embed_path else None
        if embed_dict:
            self.tt_model.pts_bbox_head.transformer.convert_torch_embeds(**embed_dict)
            self.tt_model.pts_bbox_head.convert_torch_embeds(**embed_dict)

    def run(self, data):
        self.tt_out = self.tt_model(
            rescale=True, **data, memory_config=self.memory_config, program_config=self.program_config
        )

    def run_ref(self, data):
        if self.ref_model is None:
            raise ValueError("Reference model is not initialized.")
        self.ref_out = self.ref_model(**data)

    def validate(self):
        # Compare ego future predictions between Torch and TT
        assert self.tt_out is not None, "TT output is None."
        for key in self.tt_out.keys():
            if key == "scene_query":
                continue
            _, pcc = compare_tensors(self.ref_out[key], self.tt_out[key], 0.98)
            assert pcc > 0.98, f"PCC below threshold: {pcc:.4f}; at key: {key}"
            self.logger.info("SSR test passed (PCC %.4f)", pcc)

    def dealloc_output(self):
        for v in self.tt_out.values():
            if isinstance(v, ttnn.Tensor):
                ttnn.deallocate(v)
