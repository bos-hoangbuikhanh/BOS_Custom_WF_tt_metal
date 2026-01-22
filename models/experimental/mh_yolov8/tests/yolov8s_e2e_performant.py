# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from models.bos_model.mh_yolov8.tests.yolov8s_test_infra import create_test_infra

try:
    pass

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class Yolov8sTrace2CQ:
    def __init__(self):
        ...

    def initialize_yolov8s_trace_2cqs_inference(
        self,
        device,
        device_batch_size,
        device_input_size,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
    ):
        logger.info(f"[trace_2cq] Init model and input tensor")
        self.test_infra = create_test_infra(
            device,
            device_batch_size,
            device_input_size,
        )
        self.device = device
        self.tt_inputs_host, sharded_mem_config_DRAM, self.input_mem_config = self.test_infra.setup_dram_sharded_input(
            device
        )
        self.tt_image_res = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
        self.op_event = ttnn.record_event(device, 0)

        # First run configures convs JIT
        logger.info(f"[trace_2cq] First run configures convs JIT")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        spec = self.test_infra.input_tensor.spec
        self.op_event = ttnn.record_event(device, 0)
        self.test_infra.run()
        self.test_infra.validate()
        self.test_infra.dealloc_output()

        # Optimized run
        logger.info(f"[trace_2cq] 2nd optimized run")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(device, 0)
        self.test_infra.run()
        self.test_infra.validate()

        # Capture
        logger.info(f"[trace_2cq] Capture")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(device, 0)
        self.test_infra.dealloc_output()
        trace_input_addr = self.test_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        self.test_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, device)
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        print(f"trace_input_addr: {trace_input_addr} == tt_device_tensor: {self.input_tensor.buffer_address()}")

        # assert trace_input_addr == self.input_tensor.buffer_address()
        if trace_input_addr != self.input_tensor.buffer_address():
            logger.info(f"[trace_2cq] Capture once again")
            ttnn.wait_for_event(1, self.op_event)
            ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
            self.write_event = ttnn.record_event(device, 1)
            ttnn.wait_for_event(0, self.write_event)
            self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
            self.op_event = ttnn.record_event(device, 0)
            self.test_infra.dealloc_output()
            trace_input_addr = self.test_infra.input_tensor.buffer_address()
            self.tid = ttnn.begin_trace_capture(device, cq_id=0)
            self.test_infra.run()
            self.input_tensor = ttnn.allocate_tensor_on_device(spec, device)
            ttnn.end_trace_capture(device, self.tid, cq_id=0)
            print(f"trace_input_addr: {trace_input_addr} == tt_device_tensor: {self.input_tensor.buffer_address()}")
            assert trace_input_addr == self.input_tensor.buffer_address()

    def execute_yolov8s_trace_2cqs_inference(self, tt_inputs_host=None):
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        # TODO: Add in place support to ttnn to_memory_config
        self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=True)
        # outputs = ttnn.from_device(self.test_infra.output_tensor, blocking=True)

        # return outputs
        return self.test_infra.output_tensors

    def release_yolov8s_trace_2cqs_inference(self):
        ttnn.release_trace(self.device, self.tid)


class Yolov8sTrace:
    def __init__(self):
        ...

    def initialize_yolov8s_trace_inference(
        self,
        device,
        device_batch_size,
        device_input_size,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
    ):
        logger.info(f"[trace] Init model and input tensor")
        self.test_infra = create_test_infra(
            device,
            device_batch_size,
            device_input_size,
        )
        self.device = device
        self.tt_inputs_host, self.input_mem_config = self.test_infra.setup_l1_sharded_input(device)

        # First run configures convs JIT
        logger.info(f"[trace] First run configures convs JIT")
        self.test_infra.input_tensor = self.tt_inputs_host.to(device, self.input_mem_config)
        spec = self.test_infra.input_tensor.spec
        self.test_infra.run()
        self.test_infra.validate()
        # next_input_tensor = self.tt_inputs_host.to(device, self.input_mem_config)
        # if next_input_tensor.is_allocated():
        #     ttnn.deallocate(next_input_tensor)
        self.test_infra.dealloc_output()

        # Optimized run
        logger.info(f"[trace] 2nd optimized run")
        self.test_infra.input_tensor = self.tt_inputs_host.to(device, self.input_mem_config)
        self.test_infra.run()
        self.test_infra.validate()
        # next_input_tensor = ttnn.allocate_tensor_on_device(spec, device)
        # if next_input_tensor.is_allocated():
        #     ttnn.deallocate(next_input_tensor)

        # Capture
        logger.info(f"[trace] Capture")
        self.test_infra.input_tensor = self.tt_inputs_host.to(device, self.input_mem_config)
        self.test_infra.dealloc_output()
        trace_input_addr = self.test_infra.input_tensor.buffer_address()
        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        self.test_infra.run()
        self.tt_image_res = ttnn.allocate_tensor_on_device(spec, device)
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        print(f"trace_input_addr: {trace_input_addr} == tt_device_tensor: {self.tt_image_res.buffer_address()}")

        # assert trace_input_addr == self.tt_image_res.buffer_address()
        if trace_input_addr != self.tt_image_res.buffer_address():
            logger.info(f"[trace] Capture once again")
            ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 0)
            self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
            trace_input_addr = self.test_infra.input_tensor.buffer_address()
            self.test_infra.dealloc_output()

            self.tid = ttnn.begin_trace_capture(device, cq_id=0)
            self.test_infra.run()
            self.tt_image_res = ttnn.allocate_tensor_on_device(spec, device)
            ttnn.end_trace_capture(device, self.tid, cq_id=0)
            print(f"trace_input_addr: {trace_input_addr} == tt_device_tensor: {self.tt_image_res.buffer_address()}")
            assert trace_input_addr == self.tt_image_res.buffer_address()

    def execute_yolov8s_trace_inference(self, tt_inputs_host=None):
        # tt_inputs_host = tt_inputs_host if tt_inputs_host is not None else self.tt_inputs_host
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 0)
        # TODO: Add in place support to ttnn to_memory_config
        # self.input_tensor = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.input_tensor)
        # self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=True)
        # outputs = ttnn.from_device(self.test_infra.output_tensor, blocking=True)

        # return outputs
        return self.test_infra.output_tensors

    def release_yolov8s_trace_inference(self):
        ttnn.release_trace(self.device, self.tid)
