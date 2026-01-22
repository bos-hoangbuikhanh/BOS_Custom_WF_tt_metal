import os
import time

import pytest
import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

import ttnn
from models.bos_model.vit.inference import run_vit
from models.bos_model.vit.tests.util import load_torch_model
from models.bos_model.vit.ttnn_optimized_sharded_vit_a0 import TtViT
from ttnn.core import divup

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def vit_capture_trace(pixel_values_host, vit):
    dram_grid_size = vit.device.dram_grid_size()
    pixel_values_dram = pixel_values_host.to(
        vit.device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
                ),
                [
                    divup(pixel_values_host.volume() // pixel_values_host.shape[-1], dram_grid_size.x),
                    pixel_values_host.shape[-1],
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    # initialize op events
    first_op_event = ttnn.record_event(vit.device, 0)
    read_event = ttnn.record_event(vit.device, 1)

    # 1. JIT for creation of program cache.
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(pixel_values_host, pixel_values_dram, 1)
    write_event = ttnn.record_event(vit.device, 1)
    ttnn.wait_for_event(0, write_event)
    dummy_input_l1 = ttnn.to_memory_config(pixel_values_dram, vit.input_l1_mem_config)
    first_op_event = ttnn.record_event(vit.device, 0)
    output_l1 = vit._vit_device_func(dummy_input_l1)
    output_dram = ttnn.to_memory_config(output_l1, ttnn.DRAM_MEMORY_CONFIG)
    last_op_event = ttnn.record_event(vit.device, 0)

    # 2. capture trace
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(pixel_values_host, pixel_values_dram, 1)
    write_event = ttnn.record_event(vit.device, 1)
    ttnn.wait_for_event(0, write_event)
    dummy_input_l1 = ttnn.to_memory_config(pixel_values_dram, vit.input_l1_mem_config)
    first_op_event = ttnn.record_event(vit.device, 0)

    trace_addr = dummy_input_l1.buffer_address()
    output_l1.deallocate(force=True)
    trace_id = ttnn.begin_trace_capture(vit.device, cq_id=0)
    output_l1 = vit._vit_device_func(dummy_input_l1)
    input_l1 = ttnn.allocate_tensor_on_device(dummy_input_l1.spec, vit.device)
    ttnn.end_trace_capture(vit.device, trace_id, cq_id=0)
    output_dram = ttnn.to_memory_config(output_l1, ttnn.DRAM_MEMORY_CONFIG)
    if trace_addr != input_l1.buffer_address():
        raise RuntimeError("Trace capture failed: l1 address mismatch")
    return trace_id, first_op_event, write_event, read_event, pixel_values_dram, input_l1, output_l1


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 850000, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("warmup_iters", [2])
@pytest.mark.parametrize("num_measure_iters", [4])
def test_e2e_trace2cq(device, batch_size, warmup_iters, num_measure_iters, model_location_generator):
    # setup model
    torch_vit = load_torch_model(model_location_generator)
    vit = TtViT(device, batch_size, torch_vit)

    # prepare inputs
    torch_pixel_values = torch.rand([batch_size, 3, 224, 224], dtype=torch.bfloat16)
    ttnn_pixel_values = vit._prepare_inputs(torch_pixel_values)

    trace_id, first_op_event, write_event, read_event, input_dram, input_l1, output_l1 = vit_capture_trace(
        ttnn_pixel_values, vit
    )

    # warmup
    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(ttnn_pixel_values, input_dram, 1)
    write_event = ttnn.record_event(device, 1)
    for _ in range(warmup_iters):
        ttnn.wait_for_event(0, write_event)
        input_l1 = ttnn.reshard(input_dram, vit.input_l1_mem_config, input_l1)
        first_op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)

        output_dram = ttnn.to_memory_config(output_l1, ttnn.DRAM_MEMORY_CONFIG)
        last_op_event = ttnn.record_event(device, 0)

        ttnn.wait_for_event(1, first_op_event)
        ttnn.copy_host_to_device_tensor(ttnn_pixel_values, input_dram, 1)
        write_event = ttnn.record_event(device, 1)

        ttnn.wait_for_event(1, last_op_event)
        ttnn.from_device(output_dram, blocking=True, cq_id=1)
        read_event = ttnn.record_event(device, 1)

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost(header="start")

    ttnn.wait_for_event(1, first_op_event)
    ttnn.copy_host_to_device_tensor(ttnn_pixel_values, input_dram, 1)
    write_event = ttnn.record_event(device, 1)

    outputs = []
    start = time.time()
    for _ in range(num_measure_iters):
        ttnn.wait_for_event(0, write_event)
        input_l1 = ttnn.reshard(input_dram, vit.input_l1_mem_config, input_l1)
        first_op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.wait_for_event(0, read_event)

        output_dram = ttnn.to_memory_config(output_l1, ttnn.DRAM_MEMORY_CONFIG)
        last_op_event = ttnn.record_event(device, 0)

        ttnn.wait_for_event(1, first_op_event)
        ttnn.copy_host_to_device_tensor(ttnn_pixel_values, input_dram, 1)
        write_event = ttnn.record_event(device, 1)

        ttnn.wait_for_event(1, last_op_event)
        outputs.append(ttnn.from_device(output_dram, blocking=False, cq_id=1))
        read_event = ttnn.record_event(device, 1)
    ttnn.synchronize_device(device)
    end = time.time()
    if use_signpost:
        signpost(header="stop")

    duration = end - start
    total_samples = batch_size * num_measure_iters
    logger.info(
        f"processed samples: {total_samples}, total duration: {duration:.2f}s, fps: {total_samples / duration:.2f}"
    )

    ttnn.release_trace(device, trace_id)
    return total_samples / duration


@pytest.mark.skipif(
    os.path.isdir(os.environ.get("IMAGENET-1K_VAL_DIR", "/srv/datasets/imagenet-1k")) == False,
    reason=f"skipping accuracy test as IMAGENET-1K_VAL_DIR={os.environ.get('IMAGENET-1K_VAL_DIR', '/srv/datasets/imagenet-1k')} is not available",
)
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("n_samples", [10000])
def test_accuracy(device, batch_size, n_samples, model_location_generator):
    data_dir = os.environ.get("IMAGENET-1K_VAL_DIR", "/srv/datasets/imagenet-1k")
    # setup dataloader
    preprocessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)

    def transform(samples):
        images = [preprocessor(image, return_tensors="pt")["pixel_values"][0] for image in samples["image"]]
        return {"pixel_values": images, "labels": samples["label"]}

    dataset = load_dataset("imagefolder", data_dir=data_dir, split=f"validation[:{n_samples}]").map(
        transform, batched=True, batch_size=batch_size
    )
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # setup model
    torch_vit = load_torch_model(model_location_generator)
    vit = TtViT(device, batch_size, torch_vit)

    correct = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            tt_output = vit(batch["pixel_values"])
            # accuracy
            pred = ttnn.to_torch(ttnn.from_device(tt_output)).to(torch.float)
            pred = pred[:, 0, :1000].argmax(dim=-1)
            correct += (batch["labels"] == pred).sum().item()
    logger.info(f"correct: {correct}, n_samples: {n_samples}, accuracy: {correct / n_samples * 100:.2f}%")
    # assert correct >= 8334  # 83.34% for first 10000 samples with torch.bfloat16


def vit_runner(device_id, batch_size, num_iters, **kwargs):
    assert batch_size == 5, "Only batch size 5 is supported in ViT."

    # run e2e benchmark
    device = ttnn.CreateDevice(device_id=device_id, l1_small_size=32768, trace_region_size=850000, num_command_queues=2)
    fps = test_e2e_trace2cq(
        device, batch_size, warmup_iters=15, num_measure_iters=num_iters, model_location_generator=None
    )
    ttnn.CloseDevice(device)

    # run single inference
    device = ttnn.open_device(device_id=device_id, l1_small_size=32768)
    torch_pixel_values = torch.rand([batch_size, 3, 224, 224], dtype=torch.bfloat16)
    output_tensor, _ = run_vit(batch_size, device, torch_pixel_values)
    ttnn.close_device(device)

    return {
        "fps": fps,
        "input_tensor": torch_pixel_values,
        "output_tensor": output_tensor,
    }
