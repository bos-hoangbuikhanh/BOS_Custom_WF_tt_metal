# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.bos_model.ufld_v2.runner.performant_runner_infra import UFLDPerformanceRunnerInfra
from tests.ttnn.utils_for_testing import assert_with_pcc


class UFLDPerformantRunner:
    def __init__(
        self,
        device,
        model_location_generator,
        device_batch_size=1,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        resolution=(320, 800),
        torch_input_tensor=None,
        input_mode="dram_interleaved",
        model_type="tusimple",
    ):
        self.device = device
        self.resolution = resolution
        self.torch_input_tensor = torch_input_tensor
        self.runner_infra = UFLDPerformanceRunnerInfra(
            device,
            model_location_generator,
            device_batch_size,
            act_dtype,
            weight_dtype,
            resolution=resolution,
            torch_input_tensor=self.torch_input_tensor,
            model_type=model_type,
        )

        if input_mode == "dram_sharded":
            (
                self.tt_inputs_host,
                sharded_mem_config_DRAM,
                self.dram_mem_config,
            ) = self.runner_infra.setup_dram_sharded_input(device)
            self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
            self._capture_ufldv2_trace_2cqs()
        elif input_mode == "dram_interleaved":
            self.tt_inputs_host, self.dram_mem_config = self.runner_infra.setup_dram_interleaved_input(device)
            self.tt_image_res = ttnn.allocate_tensor_on_device(
                self.tt_inputs_host.shape,
                self.tt_inputs_host.dtype,
                self.tt_inputs_host.layout,
                device,
                self.dram_mem_config,
            )
            self._capture_ufldv2_trace_2cqs_interleaved()
        # elif input_mode == "dram_interleaved_double_buffer":
        #   self.tt_inputs_host, self.dram_buffers = \
        #       self.runner_infra.setup_dram_interleaved_double_buffer(device)
        #   self.buffer_index = 0
        #   self.tt_image_res = self.dram_buffers[0]  # Initial buffer setup
        #   self._capture_ufldv2_trace_2cqs_double_buffer()
        else:
            raise ValueError(f"Unsupported input_mode: {input_mode}")

        self.input_mode = input_mode  # Store mode

    def _capture_ufldv2_trace_2cqs(self):
        # Initialize the op event so we can write
        self.op_event = ttnn.record_event(self.device, 0)
        # First run configures convs JIT
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.dram_mem_config)
        spec = self.runner_infra.input_tensor.spec
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.dram_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.dram_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        assert trace_input_addr == self.input_tensor.buffer_address()

    def _capture_ufldv2_trace_2cqs_interleaved(self):
        self.op_event = ttnn.record_event(self.device, 0)

        # First run - DRAM interleaved direct use
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.tt_image_res  # resharding removed
        spec = self.runner_infra.input_tensor.spec
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.tt_image_res
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.input_tensor = self.tt_image_res
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        # assert trace_input_addr == self.input_tensor.buffer_address()

    ## Double buffering trace capture
    # def _capture_ufldv2_trace_2cqs_double_buffer(self):
    #    self.op_event = ttnn.record_event(self.device, 0)

    #    # First run - Buffer 0
    #    ttnn.wait_for_event(1, self.op_event)
    #    ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.dram_buffers[0], 1)
    #    self.write_event = ttnn.record_event(self.device, 1)
    #    ttnn.wait_for_event(0, self.write_event)
    #    self.runner_infra.input_tensor = self.dram_buffers[0]
    #    spec = self.runner_infra.input_tensor.spec
    #    self.op_event = ttnn.record_event(self.device, 0)
    #    self.runner_infra.run()
    #    self.runner_infra.validate()
    #    self.runner_infra.dealloc_output()

    #    # Optimized run - Buffer 1
    #    ttnn.wait_for_event(1, self.op_event)
    #    ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.dram_buffers[1], 1)
    #    self.write_event = ttnn.record_event(self.device, 1)
    #    ttnn.wait_for_event(0, self.write_event)
    #    self.runner_infra.input_tensor = self.dram_buffers[1]
    #    self.op_event = ttnn.record_event(self.device, 0)
    #    self.runner_infra.run()
    #    self.runner_infra.validate()

    #    # Capture - Buffer 0 (trace compatibility)
    #    ttnn.wait_for_event(1, self.op_event)
    #    ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.dram_buffers[0], 1)
    #    self.write_event = ttnn.record_event(self.device, 1)
    #    ttnn.wait_for_event(0, self.write_event)
    #    self.runner_infra.input_tensor = self.dram_buffers[0]
    #    self.op_event = ttnn.record_event(self.device, 0)
    #    self.runner_infra.dealloc_output()
    #    trace_input_addr = self.runner_infra.input_tensor.buffer_address()
    #    self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
    #    self.runner_infra.run()
    #    self.input_tensor = ttnn.allocate_tensor_on_device(spec, self.device)
    #    ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
    #    #assert trace_input_addr == self.input_tensor.buffer_address()

    def _execute_ufldv2_trace_2cqs_inference(self, tt_inputs_host=None):
        if self.input_mode == "dram_sharded":
            return self._execute_sharded_inference(tt_inputs_host)
        elif self.input_mode == "dram_interleaved":
            return self._execute_interleaved_inference(tt_inputs_host)
        # elif self.input_mode == "dram_interleaved_double_buffer":
        #    return self._execute_double_buffer_inference(tt_inputs_host)

    def _execute_sharded_inference(self, tt_inputs_host):
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        if self.input_tensor.is_sharded():
            self.input_tensor = ttnn.reshard(self.tt_image_res, self.dram_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return self.runner_infra.output_tensor_1

    def _execute_interleaved_inference(self, tt_inputs_host):
        # DRAM interleaved single buffer logic
        tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)

        # Use DRAM interleaved directly (resharding removed)
        self.input_tensor = self.tt_image_res
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return self.runner_infra.output_tensor_1

    # def _execute_double_buffer_inference(self, tt_inputs_host):
    #   tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host

    #   # Write to current buffer
    #   current_buffer = self.dram_buffers[self.buffer_index]
    #   ttnn.wait_for_event(1, self.op_event)
    #   ttnn.copy_host_to_device_tensor(tt_inputs_host, current_buffer, 1)
    #   self.write_event = ttnn.record_event(self.device, 1)
    #   ttnn.wait_for_event(0, self.write_event)

    #   self.input_tensor = current_buffer
    #   self.op_event = ttnn.record_event(self.device, 0)
    #   ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)

    #   # Buffer toggle
    #   self.buffer_index ^= 1

    #   return self.runner_infra.output_tensor_1

    def _validate(self, input_tensor, result_output_tensor):
        torch_output_tensor = self.runner_infra.torch_output_tensor_1
        assert_with_pcc(torch_output_tensor, result_output_tensor, self.runner_infra.valid_pcc)

    def run(self, torch_input_tensor=None):
        if self.input_mode == "dram_sharded":
            tt_inputs_host, _ = self.runner_infra.setup_l1_sharded_input(self.device, torch_input_tensor)
        elif self.input_mode == "dram_interleaved":
            tt_inputs_host, _ = self.runner_infra.setup_dram_interleaved_input(self.device, torch_input_tensor)
        # elif self.input_mode == "dram_interleaved_double_buffer":
        #    tt_inputs_host, _ = self.runner_infra.setup_dram_interleaved_double_buffer(self.device, torch_input_tensor)

        output = self._execute_ufldv2_trace_2cqs_inference(tt_inputs_host)
        return output

    def release(self):
        ttnn.release_trace(self.device, self.tid)
