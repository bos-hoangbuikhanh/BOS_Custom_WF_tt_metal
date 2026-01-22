from __future__ import annotations

import os
import time
from copy import deepcopy
from test.SSR.tt.utils.misc import extract_data_from_container
from typing import Any, Dict, Optional, Tuple

import torch
from bos_metal import op, ttnn
from loguru import logger

from scripts.visualization import BEVMapLoader, Visualizer

from .performant_runner_infra import DoubleStorage, InvocationBuffer, SSRPerformanceRunnerInfra


class SSRPerformanceRunner:
    def __init__(
        self,
        device: Any,
        common_config: Optional[Any] = None,
        backbone_config: Optional[Any] = None,
        neck_config: Optional[Any] = None,
        embed_path: Optional[str] = None,
        pt_checkpoint_path: Optional[str] = None,
        tt_checkpoint_path: Optional[str] = None,
        memory_config: Optional[Any] = None,
        program_config: Optional[Any] = None,
        double_cq: bool = False,
        dataset: Optional[Any] = None,
    ) -> None:
        # Basic parameters
        self.device = device
        self.common_config = common_config

        # Compile & trace flags
        self.is_compiled = False
        self.is_captured = False
        self.trace_dict: Dict[str, Any] = {}

        # Double CQ
        self.double_cq = double_cq
        self.start_compute_event = []
        self.end_compute_event = []
        self.finish_write_event = []
        self.finish_read_event = []
        self.io_cq_id = 1 if double_cq else 0
        self.compute_cq_id = 0

        # Runner infrastructure
        self.runner_infra = SSRPerformanceRunnerInfra(
            device=self.device,
            common_config=self.common_config,
            backbone_config=backbone_config,
            neck_config=neck_config,
            embed_path=embed_path,
            pt_checkpoint_path=pt_checkpoint_path,
            tt_checkpoint_path=tt_checkpoint_path,
            memory_config=memory_config,
            program_config=program_config,
        )

        # I/O storage
        self.input_storage = DoubleStorage() if self.double_cq else None
        self.output_storage: Dict[str, Any] = {}
        self.host_output: Dict[str, Any] = {}

        # Scheduler
        self.sched = InvocationBuffer(device=self.device, target_cq_id=self.io_cq_id)

        # Post-process helpers
        self.dataset = dataset
        self.visualizer: Optional[Visualizer] = None
        self.bevloader: Optional[BEVMapLoader] = None

        # Timing
        self.execution_time = 0.0
        self.last_time = time.time()

    # ----------------------------- Small utilities ---------------------------

    def _event_synchronize(self, event_attr: str) -> None:
        """Synchronize on the first event in the list attribute `event_attr`."""
        ttnn.event_synchronize(getattr(self, event_attr)[0])

    def _record_event(self, event_attr: str, cq_id: int) -> None:
        """Record an event onto the list attribute `event_attr`."""
        getattr(self, event_attr).append(ttnn.record_event(self.device, cq_id))

    def _record_execution_time(self) -> None:
        """Update self.execution_time from the last timestamp and move the cursor."""
        end_time = time.time()
        self.execution_time = end_time - getattr(self, "last_time", 0.0)
        self.last_time = end_time

    def _copy_to_host(self, *args, device: Optional[Any] = None, cq_id: int = 0) -> None:
        """
        Copy device outputs to host tensors.
        NOTE: Accepts *args to tolerate older call sites that accidentally passed
        an extra first positional argument. It is ignored to preserve behavior.
        """
        for key in self.output_storage:
            self.host_output[key] = ttnn.to_torch(tensor=self.output_storage[key], device=device, cq_id=cq_id)

    def _clone_tt_out_to_output_storage(self) -> None:
        """Clone current tt_out to persistent output_storage."""
        for key in self.runner_infra.tt_out:
            self.output_storage[key] = ttnn.clone(self.runner_infra.tt_out[key])

    def _persist_tt_out_to_output_storage(self) -> None:
        """Identity-copy current tt_out to persistent output_storage (in-place reuse)."""
        for key in self.runner_infra.tt_out:
            self.output_storage[key] = ttnn.identity(
                input_tensor=self.runner_infra.tt_out[key],
                memory_config=self.runner_infra.tt_out[key].memory_config(),
                output_tensor=self.output_storage[key],
            )

    def _should_post_process(self, **kwargs: Any) -> bool:
        return kwargs.get("visualize", False) or kwargs.get("validate", False)

    # ----------------------------- Core modes --------------------------------

    def _compile_run(self, data: Dict[str, Any]) -> None:
        """Compile-only path; sets `is_compiled` when done."""
        if self.double_cq:
            # Dummy record op-event and read-event to satisfy run loop ordering
            self._record_event("start_compute_event", self.compute_cq_id)
            self._record_event("finish_read_event", self.io_cq_id)

            for _ in range(2):
                input_data = deepcopy(data)
                ttnn.wait_for_event(self.io_cq_id, self.start_compute_event.pop(0))
                self.input_storage.set(
                    extract_data_from_container(
                        data=input_data,
                        tensor="tt",
                        device=self.device,
                        input_config=self.common_config["input_config"],
                        cq_id=self.io_cq_id,
                    )
                )
                self._record_event("finish_write_event", self.io_cq_id)

                # Compute CQ waits for write CQ
                ttnn.wait_for_event(self.compute_cq_id, self.finish_write_event.pop(0))
                self._record_event("start_compute_event", self.compute_cq_id)
                self.runner_infra.run(self.input_storage.get())

                # Persist to device storage
                ttnn.wait_for_event(self.compute_cq_id, self.finish_read_event.pop(0))
                self._clone_tt_out_to_output_storage()
                self._record_event("end_compute_event", self.compute_cq_id)

                ttnn.wait_for_event(self.io_cq_id, self.end_compute_event.pop(0))
                self._copy_to_host(device=self.device, cq_id=self.io_cq_id)
                self._record_event("finish_read_event", self.io_cq_id)
                self._event_synchronize("finish_read_event")
        else:
            # Single CQ
            self.input_storage = extract_data_from_container(
                data=deepcopy(data),
                tensor="tt",
                device=self.device,
                input_config=self.common_config["input_config"],
                cq_id=self.io_cq_id,
            )
            self.runner_infra.run(self.input_storage)

            # Persist and host read
            self._clone_tt_out_to_output_storage()
            self._copy_to_host(self.output_storage, device=self.device, cq_id=self.io_cq_id)

            ttnn.synchronize_device(self.device)

        self.is_compiled = True

    def _reference_run(self, data: Dict[str, Any]) -> Any:
        """Run PyTorch reference (no gradients)."""
        data_pt = extract_data_from_container(deepcopy(data), tensor="pt")
        with torch.no_grad():
            self.runner_infra.run_ref(data_pt)

        _, bbox_results = self.simple_test(
            outs=self.runner_infra.ref_out,
            img_metas=(data["img_metas"][0].data)[0],
            ego_fut_cmd=(data["ego_fut_cmd"][0].data)[0],
        )
        return bbox_results

    def _normal_run(self, data: Dict[str, Any], **kwargs: Any) -> Tuple[Dict[str, Any], float]:
        """Normal path (optionally visualize and/or compute metrics)."""
        if self.double_cq:
            input_data = deepcopy(data)
            with self.sched.add_block(priority=0) as defer:
                defer(lambda: ttnn.wait_for_event(self.io_cq_id, self.start_compute_event.pop(0)))
                defer(
                    lambda: self.input_storage.set(
                        extract_data_from_container(
                            data=input_data,
                            tensor="tt",
                            output_storage=self.input_storage.get(mode="write"),
                            device=self.device,
                            input_config=self.common_config["input_config"],
                            cq_id=self.io_cq_id,
                        )
                    )
                )
                defer(lambda: self._record_event("finish_write_event", self.io_cq_id))

            # Flush: write input n, read output n-1
            self.sched.flush(num_blocks=2)

            # Compute CQ waits for write CQ
            ttnn.wait_for_event(self.compute_cq_id, self.finish_write_event.pop(0))
            self._record_event("start_compute_event", self.compute_cq_id)
            self.runner_infra.run(self.input_storage.get())

            # Flush post-process of output n-1
            self.sched.flush()

            # Persist to device storage
            ttnn.wait_for_event(self.compute_cq_id, self.finish_read_event.pop(0))
            self._persist_tt_out_to_output_storage()
            self._record_event("end_compute_event", self.compute_cq_id)

            # Final sample branch
            if kwargs.get("sample_idx", 0) == len(self.dataset) - 1 or kwargs.get("sample_idx", 0) == -1:
                ttnn.wait_for_event(self.io_cq_id, self.end_compute_event.pop(0))
                self._copy_to_host(device=self.device, cq_id=self.io_cq_id)
                self._record_event("finish_read_event", self.io_cq_id)
                self._event_synchronize("finish_read_event")
                if self._should_post_process(**kwargs):
                    self._post_process(self.host_output, data, **kwargs)
            else:
                # Schedule host copy and post steps
                with self.sched.add_block(priority=1) as defer:
                    defer(lambda: ttnn.wait_for_event(self.io_cq_id, self.end_compute_event.pop(0)))
                    defer(lambda: self._copy_to_host(device=self.device, cq_id=self.io_cq_id))
                    defer(lambda: self._record_event("finish_read_event", self.io_cq_id))
                with self.sched.add_block(priority=2) as defer:
                    defer(lambda: self._event_synchronize("finish_read_event"))
                    defer(lambda: self._record_execution_time())
                    if not kwargs.get("visualize", False):
                        defer(
                            lambda: logger.debug(
                                f"Sample {kwargs.get('sample_idx', 0)} processed in "
                                f"{self.execution_time:.4f} seconds -- FPS: {1.0 / self.execution_time:.2f}"
                            )
                        )
                    if self._should_post_process(**kwargs):
                        defer(lambda: self._post_process(self.host_output, data, **kwargs))
        else:
            # Single CQ
            self.input_storage = extract_data_from_container(
                data=deepcopy(data),
                output_storage=self.input_storage,
                tensor="tt",
                device=self.device,
                input_config=self.common_config["input_config"],
                cq_id=self.io_cq_id,
            )
            self.runner_infra.run(self.input_storage)

            # Persist and host read
            self._persist_tt_out_to_output_storage()
            self._copy_to_host(self.output_storage, device=self.device, cq_id=self.io_cq_id)

            # Post
            ttnn.synchronize_device(self.device)
            self._record_execution_time()
            if not kwargs.get("visualize", False):
                logger.debug(
                    f"Sample {kwargs.get('sample_idx', 0)} processed in "
                    f"{self.execution_time:.4f} seconds -- FPS: {1.0 / self.execution_time:.2f}"
                )
            if self._should_post_process(**kwargs):
                self._post_process(self.host_output, data, **kwargs)

        return self.host_output, self.execution_time

    def _capture_trace(self, data: Dict[str, Any]) -> None:
        """Capture trace(s) for later replay; sets `is_captured`."""
        ttnn.synchronize_device(self.device)

        if self.double_cq:
            for _ in range(3):
                ttnn.synchronize_device(self.device)
                trace_idx = self.input_storage.write_idx
                self.input_storage.set(
                    extract_data_from_container(
                        data=deepcopy(data),
                        tensor="tt",
                        output_storage=self.input_storage.get(mode="write"),
                        device=self.device,
                        input_config=self.common_config["input_config"],
                        cq_id=self.io_cq_id,
                    )
                )
                ttnn.synchronize_device(self.device)
                if self.trace_dict.get(f"trace_{trace_idx}", None) is not None:
                    ttnn.release_trace(self.device, self.trace_dict[f"trace_{trace_idx}"])
                self.trace_dict[f"trace_{trace_idx}"] = ttnn.begin_trace_capture(self.device, cq_id=self.compute_cq_id)
                self.runner_infra.run(self.input_storage.get())
                ttnn.end_trace_capture(self.device, self.trace_dict[f"trace_{trace_idx}"], cq_id=self.compute_cq_id)
                ttnn.synchronize_device(self.device)
        else:
            ttnn.synchronize_device(self.device)
            self.input_storage = extract_data_from_container(
                data=deepcopy(data),
                output_storage=self.input_storage,
                tensor="tt",
                device=self.device,
                input_config=self.common_config["input_config"],
                cq_id=self.io_cq_id,
            )
            ttnn.synchronize_device(self.device)
            self.trace_dict["trace_0"] = ttnn.begin_trace_capture(self.device, cq_id=self.compute_cq_id)
            self.runner_infra.run(self.input_storage)
            ttnn.end_trace_capture(self.device, self.trace_dict["trace_0"], cq_id=self.compute_cq_id)
            ttnn.synchronize_device(self.device)

        self.is_captured = True

    def _execute_trace(self, data: Dict[str, Any], **kwargs: Any) -> Tuple[Dict[str, Any], float]:
        """Execute previously captured trace (performant replay)."""
        if self.double_cq:
            trace_idx = self.input_storage.write_idx
            input_data = deepcopy(data)

            with self.sched.add_block(priority=0) as defer:
                defer(lambda: ttnn.wait_for_event(self.io_cq_id, self.start_compute_event.pop(0)))
                defer(
                    lambda: self.input_storage.set(
                        extract_data_from_container(
                            data=input_data,
                            tensor="tt",
                            output_storage=self.input_storage.get(mode="write"),
                            device=self.device,
                            input_config=self.common_config["input_config"],
                            cq_id=self.io_cq_id,
                        )
                    )
                )
                defer(lambda: self._record_event("finish_write_event", self.io_cq_id))

            # Flush: write input n, read output n-1
            self.sched.flush(num_blocks=2)

            # Compute CQ waits for write CQ
            ttnn.wait_for_event(self.compute_cq_id, self.finish_write_event.pop(0))
            self._record_event("start_compute_event", self.compute_cq_id)
            ttnn.execute_trace(
                self.device, self.trace_dict[f"trace_{trace_idx}"], cq_id=self.compute_cq_id, blocking=False
            )

            # Flush post-process of output n-1
            self.sched.flush()

            # Persist & mark end
            ttnn.wait_for_event(self.compute_cq_id, self.finish_read_event.pop(0))
            self._persist_tt_out_to_output_storage()
            self._record_event("end_compute_event", self.compute_cq_id)

            # Final sample branch
            if kwargs.get("sample_idx", 0) == len(self.dataset) - 1 or kwargs.get("sample_idx", 0) == -1:
                ttnn.wait_for_event(self.io_cq_id, self.end_compute_event.pop(0))
                self._copy_to_host(device=self.device, cq_id=self.io_cq_id)
                self._record_event("finish_read_event", self.io_cq_id)
                self._event_synchronize("finish_read_event")
                if self._should_post_process(**kwargs):
                    self._post_process(self.host_output, data, **kwargs)
            else:
                with self.sched.add_block(priority=1) as defer:
                    defer(lambda: ttnn.wait_for_event(self.io_cq_id, self.end_compute_event.pop(0)))
                    defer(lambda: self._copy_to_host(device=self.device, cq_id=self.io_cq_id))
                    defer(lambda: self._record_event("finish_read_event", self.io_cq_id))
                with self.sched.add_block(priority=2) as defer:
                    defer(lambda: self._event_synchronize("finish_read_event"))
                    defer(lambda: self._record_execution_time())
                    if not kwargs.get("visualize", False):
                        defer(
                            lambda: logger.debug(
                                f"Sample {kwargs.get('sample_idx', 0)} processed in "
                                f"{self.execution_time:.4f} seconds -- FPS: {1.0 / self.execution_time:.2f}"
                            )
                        )
                    if self._should_post_process(**kwargs):
                        defer(lambda: self._post_process(self.host_output, data, **kwargs))
        else:
            # Single CQ
            self.input_storage = extract_data_from_container(
                data=deepcopy(data),
                output_storage=self.input_storage,
                tensor="tt",
                device=self.device,
                input_config=self.common_config["input_config"],
                cq_id=self.io_cq_id,
            )
            ttnn.execute_trace(self.device, self.trace_dict["trace_0"], cq_id=self.io_cq_id, blocking=False)

            # Persist & host read
            self._persist_tt_out_to_output_storage()
            self._copy_to_host(device=self.device, cq_id=self.io_cq_id)

            # Post
            ttnn.synchronize_device(self.device)
            self._record_execution_time()
            if not kwargs.get("visualize", False):
                logger.debug(
                    f"Sample {kwargs.get('sample_idx', 0)} processed in "
                    f"{self.execution_time:.4f} seconds -- FPS: {1.0 / self.execution_time:.2f}"
                )
            if self._should_post_process(**kwargs):
                self._post_process(self.host_output, data, **kwargs)

        return self.host_output, self.execution_time

    # ----------------------------- Paths & Post -------------------------------

    def get_output_path(self, out_dir: str = "generated/video") -> str:
        """Construct a dated mp4 path under `out_dir` (absolute if needed)."""
        current_path = os.getcwd()
        current_path = current_path.rsplit("/", 1)[0] + "/"
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(current_path, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(out_dir, f"realtime_{ts}.mp4")

    def _post_process(self, tt_result: Dict[str, Any], data: Dict[str, Any], **kwargs: Any) -> None:
        """Visualization/validation path (batch size currently restricted to 1)."""
        _, bbox_results = self.simple_test(
            outs=tt_result,
            img_metas=(data["img_metas"][0].data)[0],
            ego_fut_cmd=(data["ego_fut_cmd"][0].data)[0],
        )
        tt_result = bbox_results

        if kwargs.get("visualize", False) and kwargs.get("use_bev", False):
            st_bev = time.time()
            sample = self.bevloader.get_batch_bev_map_info(kwargs.get("sample_idx", 0))
            self.bevloader.draw_batch_bev_map(sample)
            logger.debug(f"Done draw BEV Map, taken: {time.time() - st_bev:.4f}")

        if kwargs.get("visualize", False):
            v_st = time.time()
            visualize_result = self.dataset._format_bbox_realtime(
                {"bbox_results": tt_result}["bbox_results"], is_ttnn=True, sample_idx=kwargs.get("sample_idx", 0)
            )
            self.visualizer.create_visual(
                index=kwargs.get("sample_idx", 0),
                visualize_result=visualize_result,
                car_path="../scripts/car_visualize.png",
                fps=round(1.0 / self.execution_time, 2),
            )
            if kwargs.get("use_bev", False):
                visualize_img = self.visualizer.get_visual()
                self.bevloader.get_token_id(visualize_result)
                new_visual = self.bevloader.create_output_bev_map(canvas=visualize_img)
                self.visualizer.set_visual(new_visual)

            if kwargs.get("realtime", False):
                keep_running = self.visualizer.show_realtime()
                if not keep_running:
                    logger.info("Realtime display stopped by user.")
            else:
                self.visualizer.show_video()

        if kwargs.get("validate", False):
            pt_result = self._reference_run(data)
            tt_fut_preds = pt_result[0]["pts_bbox"]["ego_fut_preds"]
            pt_fut_preds = tt_result[0]["pts_bbox"]["ego_fut_preds"]
            _passed, _msg = op.compare_tensors(pt_fut_preds, tt_fut_preds, pcc=kwargs.get("pcc_threshold", 0.97))

    # ----------------------------- Public API --------------------------------

    def dealloc_output(self) -> None:
        self.runner_infra.dealloc_output()

    def close_visualizer(self, **kwargs: Any) -> None:
        if kwargs.get("realtime", False):
            self.visualizer.close_realtime()
            logger.info("Close window")
        else:
            self.visualizer.close_video()
            logger.info(f"Video saved at {self.visualizer.video_path}")

    def simple_test(
        self,
        outs: Dict[str, Any],
        img_metas: Any,
        ego_fut_cmd: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Test function without augmentation."""
        bbox_list = [dict() for _ in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(outs, ego_fut_cmd=ego_fut_cmd)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
            result_dict["metric_results"] = metric_dict
        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        outs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any, Any]:
        """Test function (batch_size=1 only)."""
        bbox_results = []
        for i in range(len(outs["ego_fut_preds"])):
            bbox_result = dict()
            bbox_result["ego_fut_preds"] = outs["ego_fut_preds"][i]
            bbox_result["ego_fut_cmd"] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, "only support batch_size=1 now"
        metric_dict = None
        return outs["bev_embed"], bbox_results, metric_dict

    def run(self, data: Dict[str, Any], mode: str = "performant", **kwargs: Any) -> Any:
        # Lazy-init visualization utilities if needed
        if kwargs.get("visualize", False):
            if self.visualizer is None:
                self.visualizer = Visualizer(dataset=self.dataset)
            if not kwargs.get("realtime", False):
                self.visualizer.set_output(self.get_output_path())
            if kwargs.get("use_bev", False) and self.bevloader is None:
                self.bevloader = BEVMapLoader(dataset=self.dataset)

        output = None
        if mode == "reference":
            output = self._reference_run(data)
        elif mode == "compile":
            self._compile_run(data)
        elif mode == "normal":
            if not self.is_compiled:
                raise RuntimeError("Not compiled. Please run in 'compile' mode first.")
            output = self._normal_run(data, **kwargs)
        elif mode == "trace_capture":
            if not self.is_compiled:
                raise RuntimeError("Not compiled. Please run in 'compile' mode first.")
            if self.is_captured:
                raise RuntimeError("Trace already captured. Please run in 'performant' mode.")
            self._capture_trace(data)
        elif mode == "performant":
            if not self.is_compiled:
                raise RuntimeError("Not compiled. Please run in 'compile' mode first.")
            if not self.is_captured:
                raise RuntimeError("Trace not captured. Please run in 'capture' mode first.")
            output = self._execute_trace(data, **kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return output

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def release(self) -> None:
        for trace in self.trace_dict:
            if self.trace_dict[trace] is None:
                continue
            ttnn.release_trace(self.device, self.trace_dict[trace])
