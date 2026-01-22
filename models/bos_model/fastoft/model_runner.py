# for display
import argparse
import os

import torch

os.environ["LOGURU_LEVEL"] = "INFO"
from loguru import logger

import ttnn
from conftest import model_location_generator

# Import OFT components
from models.bos_model.fastoft.reference.oftnet import FrontendMode, OftMode, OftNet
from models.bos_model.fastoft.reference.utils import load_calib, load_image, make_grid
from models.bos_model.fastoft.tests.common import (
    GRID_HEIGHT,
    GRID_RES,
    GRID_SIZE,
    H_PADDED,
    NMS_THRESH,
    W_PADDED,
    Y_OFFSET,
    load_checkpoint,
)
from models.bos_model.fastoft.tests.test_perf_e2e_oft import OFTPerformanceRunnerInfra
from models.bos_model.fastoft.tt.oft_pipeline import PipelineConfig, create_pipeline_from_config
from models.common.utility_functions import profiler


def fastoft_runner(device_id, batch_size, num_iter, **kwargs):
    assert batch_size == 1, "Only batch size 1 is supported in fastOFT demo"
    num_iter = num_iter if num_iter >= 0 else float("inf")

    l1_small_size = kwargs.get("l1_small_size", 32768)
    num_command_queues = kwargs.get("num_command_queues", 2)
    trace_region_size = kwargs.get("trace_region_size", 3000000)
    enable_trace = kwargs.get("enable_trace", True)
    input_image_path = kwargs.get("input_image_path", "models/bos_model/fastoft/resources/000013.jpg")
    calib_text_path = kwargs.get("calib_text_path", "models/bos_model/fastoft/resources/000013.txt")
    num_warmup_iterations = kwargs.get("num_warmup_iterations", 0)
    weight_path = kwargs.get("weight_path", None)
    preallocate_maximum_iterations = kwargs.get("preallocate_maximum_iterations", 1000)

    device = ttnn.CreateDevice(
        device_id=device_id,
        l1_small_size=l1_small_size,
        num_command_queues=num_command_queues,
        trace_region_size=trace_region_size,
    )

    model_dtype = torch.float32
    oft_mode = OftMode.OFT8
    frontend_mode = FrontendMode.REDUCED

    profiler.clear()
    device.disable_and_clear_program_cache()

    # ========================================================
    # OFT model configuration based on real model parameters
    logger.info(f"** Generating Torch ref model **")

    # 1 Handle inputs
    logger.info(f"** Loading {num_command_queues} input images for round-robin testing **")
    input_tensors = []
    for _ in range(num_command_queues):
        try:
            img_tensor, _, _ = load_image(input_image_path, pad_hw=(H_PADDED, W_PADDED), dtype=model_dtype)
            img_tensor = img_tensor[None].to(model_dtype)
        except:
            img_tensor = torch.rand((3, H_PADDED, W_PADDED), dtype=model_dtype)[None].to(model_dtype)
        input_tensors.append(img_tensor)

    # Use first calib/grid for model setup (all should be compatible)
    try:
        calib = load_calib(calib_text_path, dtype=model_dtype)[None].to(model_dtype)
    except:
        calib = torch.randn((3, 4), dtype=model_dtype)[None].to(model_dtype)
    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=model_dtype)[None].to(model_dtype)

    # 2 Create reference OFTnet
    logger.info("** Creating reference model **")
    ref_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        dtype=model_dtype,
        oft_mode=oft_mode,
        frontend_mode=frontend_mode,
    )

    if weight_path is not None:
        os.environ["CHECKPOINTS_PATH"] = weight_path
    elif os.environ.get("CHECKPOINTS_PATH") is None:
        os.environ["CHECKPOINTS_PATH"] = "models/bos_model/fastoft/resources/checkpoint-best-no-dist_01.pth.gz"
    ref_model = load_checkpoint(ref_model, model_location_generator, oft_mode=oft_mode)
    ref_model.eval()

    # Create test infrastructure (use first input for setup)
    logger.info("** Creating ttnn test infrastructure **")
    device.enable_program_cache()
    test_infra = OFTPerformanceRunnerInfra(
        device, input_tensors[0], calib, grid, ref_model, oft_mode, model_location_generator
    )

    ttnn.synchronize_device(device)

    torch_outputs = []
    sum_inference_time = 0

    try:
        with torch.no_grad():
            # Use first input for warmup and compilation
            test_infra.input_tensor_host = input_tensors[0]

            # Setup DRAM sharded input
            tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)

            # Create pipeline
            pipeline = create_pipeline_from_config(
                config=PipelineConfig(
                    use_trace=enable_trace,
                    num_command_queues=num_command_queues,
                    all_transfers_on_separate_command_queue=False,
                ),
                model=test_infra,
                device=device,
                dram_input_memory_config=sharded_mem_config_DRAM,
                l1_input_memory_config=None,
            )

            logger.info(f"** Compiling model **")
            profiler.start("compile")
            pipeline.compile(tt_inputs_host)
            profiler.end("compile")
            logger.info(f"** Compiling model complete **")

            # Run warmup iterations with first input only
            if num_warmup_iterations > 0:
                logger.info(f"** Warmup {num_warmup_iterations} iterations **")
                warmup_inputs = [tt_inputs_host] * num_warmup_iterations
                profiler.start("warmup")
                pipeline.preallocate_output_tensors_on_host(num_warmup_iterations)
                pipeline.enqueue(warmup_inputs).pop_all()
                profiler.end("warmup")

            if num_iter > preallocate_maximum_iterations:
                logger.info(
                    f"** Since the num_iter is too large to pre-allocate input & output tensors, so we divide the iterations into chunks of size {preallocate_maximum_iterations} **"
                )
            iter_chunk_size = min(preallocate_maximum_iterations, num_iter)
            num_iter_chunk = (
                (num_iter + iter_chunk_size - 1) // iter_chunk_size if num_iter != float("inf") else num_iter
            )
            iter_chunk_idx = 0
            while iter_chunk_idx < num_iter_chunk:
                if iter_chunk_idx == num_iter_chunk - 1:
                    preallocate_size = num_iter - iter_chunk_idx * iter_chunk_size
                else:
                    preallocate_size = iter_chunk_size

                host_inputs = []
                for i in range(preallocate_size):
                    input_idx = i % len(input_tensors)
                    test_infra.input_tensor_host = input_tensors[input_idx]
                    tt_input_host, _, _ = test_infra.setup_dram_sharded_input(device)
                    host_inputs.append(tt_input_host)

                # Preallocate outputs
                pipeline.preallocate_output_tensors_on_host(preallocate_size)

                logger.info(
                    f"Starting performance pipeline for {iter_chunk_idx * iter_chunk_size}-{iter_chunk_idx * iter_chunk_size + preallocate_size} iterations"
                )
                profiler.start(f"run_model_pipeline_{num_command_queues}cqs")
                outputs = pipeline.enqueue(host_inputs).pop_all()
                profiler.end(f"run_model_pipeline_{num_command_queues}cqs", PERF_CNT=preallocate_size)
                sum_inference_time += profiler.get(f"run_model_pipeline_{num_command_queues}cqs") * preallocate_size

                # postprocess
                postprocess_outputs = []
                for i, output in enumerate(outputs):
                    torch_outputs = test_infra.postprocess(output)
                    postprocess_outputs.append(torch_outputs)
                    # tt_objects_torch = test_infra.tt_encoder.create_objects(*torch_outputs)
                    # postprocess_outputs.append(tt_objects_torch)
                iter_chunk_idx += 1

            pipeline.cleanup()
    finally:
        ttnn.close_device(device)

        compile_time = profiler.get("compile")
        warmup_time = profiler.get("warmup")
        avg_inference_time = profiler.get(f"run_model_pipeline_{num_command_queues}cqs")
        num_of_iters = iter_chunk_size * (iter_chunk_idx - 1) + preallocate_size
        total_images_processed = batch_size * (num_of_iters if num_iter == float("inf") else num_iter)
        avg_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
        logger.info("=" * 50)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Number of command queues            : {num_command_queues}")
        logger.info(f"Batch size                          : {batch_size}")
        logger.info(f"Number of iteration                 : {num_of_iters}")
        logger.info(f"Total images processed              : {total_images_processed}")
        logger.info(f"Compile time                        : {compile_time:.4f} s")
        logger.info(f"Warmup iterations                   : {num_warmup_iterations}")
        logger.info(f"Sum of warmup time                  : {warmup_time:.4f} s")
        logger.info(
            f"Average warmup time per iter        : {warmup_time / num_warmup_iterations if num_warmup_iterations > 0 else 0:.4f} s"
        )
        logger.info(f"Sum of inference time               : {sum_inference_time:.4f} s")
        logger.info(f"Average inference time per iter     : {avg_inference_time:.4f} s")
        logger.info(f"Average FPS                         : {avg_fps:.4f}")
        logger.info("=" * 50)

        return {
            "fps": avg_fps,
            "output_tensor": list(postprocess_outputs[-1]),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0, help="device id to use for inference")
    parser.add_argument("--batch", type=int, default=1, help="Size of batch")
    parser.add_argument("--num_iter", type=int, default=10, help="number of iteration to process. -1 is infinite loop")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "resources/000013.jpg"),
        help="input image path",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "resources/000013.txt"),
        help="calibration file path",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "resources/checkpoint-best-no-dist_01.pth.gz"),
        help="model weight path",
    )

    args = parser.parse_args()

    cfg = {
        "l1_small_size": 32768,
        "num_command_queues": 2,
        "trace_region_size": 3000000,
        "enable_trace": True,
        "input_image_path": args.input,
        "calib_text_path": args.calib,
        "num_warmup_iterations": 2,
        "weight_path": args.weight,
        "preallocate_maximum_iterations": 100,
    }
    fastoft_runner(args.device_id, args.batch, args.num_iter, **cfg)
