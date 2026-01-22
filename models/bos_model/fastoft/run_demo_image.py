# for display
import argparse
import multiprocessing

# import matplotlib
# matplotlib.use("QtAgg")
import cv2
import time

import os
import torch
# import matplotlib.pyplot as plt
from loguru import logger
logger.remove()  # Removes the default stdout sink

import ttnn
from models.common.utility_functions import profiler

# Import OFT components
from models.bos_model.fastoft.reference.oftnet import FrontendMode, OftMode, OftNet
from models.bos_model.fastoft.reference.utils import load_calib, load_image, make_grid
from models.bos_model.fastoft.tests.common import GRID_HEIGHT, GRID_RES, GRID_SIZE, H_PADDED, NMS_THRESH, W_PADDED, Y_OFFSET, load_checkpoint
from models.bos_model.fastoft.tt.oft_pipeline import PipelineConfig, create_pipeline_from_config
from models.bos_model.fastoft.reference.bbox import visualize_objects_cv

def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", "--num_iter", type=int, default=1, help="number of iterations to process. -1 is infinite loop"
    )
    parser.add_argument("-i", "--input", type=str, default=os.path.join(os.path.dirname(__file__), "resources"), help="input images directory or a single image path")
    parser.add_argument("-c", "--calib", type=str, default=os.path.join(os.path.dirname(__file__), "resources"), help="calibration file directory or a single calibration file path")
    parser.add_argument(
        "--prep",
        choices=["padding", "stretch", "crop_top", "crop_bottom", "crop_center"],
        default="padding",
        help="Input preprocessing mode (default: padding)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (single input) or directory (multi input) for saved images",
    )
    parser.add_argument(
        "--crop_k",
        type=float,
        default=0,
        help="Adjustment crop pixel y-offset when using crop",
    )
    parser.add_argument("--full_res", action="store_true", help="Use full resolution image for display")

    args, _ = parser.parse_known_args(argv)
    return args

def cv2_worker(lock, shared_dict):
    while shared_dict["running"]:
        with lock:
            output_image = shared_dict["output_image"]
        if output_image is not None:
            cv2.imshow("OFT", output_image)
            if not shared_dict["running"]:  # run all images
                key = cv2.waitKey(0)
            else:
                key = cv2.waitKey(10)
            if key == 27 or key == ord("q"):  # ESC or 'q'
                shared_dict["running"] = False
                break
    cv2.destroyAllWindows()

from models.bos_model.fastoft.tests.test_perf_e2e_oft import OFTPerformanceRunnerInfra

def main(device, model_location_generator, input_image_paths, calib_paths, args):
    model_dtype=torch.float32
    oft_mode = OftMode.OFT8
    frontend_mode = FrontendMode.REDUCED

    profiler.clear()
    device.disable_and_clear_program_cache()

    # Start inference thread
    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    shared_dict = manager.dict(
        {
            "running": True,
            "output_image": None,
        }
    )
    cv2_thread = multiprocessing.Process(target=cv2_worker, args=(lock, shared_dict))

    # ========================================================
    # OFT model configuration based on real model parameters
    print(f"** Generating Torch ref model **")

    # 1 Handle inputs
    print(f"** Loading {len(input_image_paths)} input images for round-robin testing **")
    input_tensors = []
    prep_configs = []
    orig_images = []
    for img_path in input_image_paths:
        img_tensor, orig_image, prep_config = load_image(
            img_path, pad_hw=(H_PADDED, W_PADDED), dtype=model_dtype, prep=args.prep, crop_k=args.crop_k
        )
        img_tensor = img_tensor[None].to(model_dtype)
        input_tensors.append(img_tensor)
        orig_images.append(orig_image)
        prep_configs.append(prep_config)

    # Use first calib/grid for model setup (all should be compatible)
    calib = load_calib(calib_paths[0], dtype=model_dtype)[None].to(model_dtype)
    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=model_dtype)[None].to(
        model_dtype
    )

    # 2 Create reference OFTnet
    print("** Creating reference model **")
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
    ref_model = load_checkpoint(ref_model, model_location_generator, oft_mode=oft_mode)
    ref_model.eval()

    # Create test infrastructure (use first input for setup)
    print("** Creating ttnn test infrastructure **")
    device.enable_program_cache()
    test_infra = OFTPerformanceRunnerInfra(
        device, input_tensors[0], calib, grid, ref_model, oft_mode, model_location_generator
    )

    ttnn.synchronize_device(device)

    with torch.no_grad():
        # _run_model_pipeline with 2cqs and trace
        _num_command_queues = 1
        _trace = True

        # Use first input for warmup and compilation
        test_infra.input_tensor_host = input_tensors[0]

        # Setup DRAM sharded input
        tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)

        # Create pipeline
        pipeline = create_pipeline_from_config(
            config=PipelineConfig(
                use_trace=_trace, num_command_queues=_num_command_queues, all_transfers_on_separate_command_queue=False
            ),
            model=test_infra,
            device=device,
            dram_input_memory_config=sharded_mem_config_DRAM,
            l1_input_memory_config=None,
        )

        print(f"** Compiling model **")
        profiler.start("compile")
        pipeline.compile(tt_inputs_host)
        profiler.end("compile")
        print(f"** Compiling model complete **")

        num_iter = args.num_iter if args.num_iter >= 0 else float("inf")

        num_warmup_iterations = 5 if num_iter > 1 else 0
        # Run warmup iterations with first input only
        if num_warmup_iterations > 0:
            print(f"** Warmup {num_warmup_iterations} iterations **")
            warmup_inputs = [tt_inputs_host] * num_warmup_iterations
            pipeline.preallocate_output_tensors_on_host(num_warmup_iterations)
            pipeline.enqueue(warmup_inputs).pop_all()
            print(f"** Warmup {num_warmup_iterations} iterations complete **")

        iter = 0
        command_queues_preload_size = 1
        cv2_thread.start()
        while True:
            org_inputs = []
            resized_inputs = []
            org_input_paths = []
            org_prep_configs = []
            host_inputs = []
            for _ in range(command_queues_preload_size):
                input_idx = iter % len(input_tensors)
                test_infra.input_tensor_host = input_tensors[input_idx]
                org_inputs.append(orig_images[input_idx])
                resized_inputs.append(input_tensors[input_idx])
                org_input_paths.append(input_image_paths[input_idx])
                org_prep_configs.append(prep_configs[input_idx])
                tt_input_host, _, _ = test_infra.setup_dram_sharded_input(device)
                host_inputs.append(tt_input_host)

            # Preallocate outputs
            pipeline.preallocate_output_tensors_on_host(command_queues_preload_size)

            logger.info(f"Starting performance pipeline for {command_queues_preload_size} iterations")
            profiler.start(f"run_model_pipeline_{_num_command_queues}cqs")
            outputs = pipeline.enqueue(host_inputs).pop_all()
            profiler.end(f"run_model_pipeline_{_num_command_queues}cqs")
            iter += 1
            inference_time = profiler.get(f"run_model_pipeline_{_num_command_queues}cqs")
            print(f"[{iter}] images processed avg FPS: {(command_queues_preload_size/inference_time):.2f} Hz")

            # Visualize predictions
            drawing_inputs = org_inputs if args.full_res else resized_inputs
            for i, (_input, _output) in enumerate(zip(drawing_inputs,outputs)):
                torch_outputs = test_infra.postprocess(_output)
                tt_objects_torch = test_infra.tt_encoder.create_objects(*torch_outputs)
                _input = _input.to(torch.float32).squeeze(0)
                calib = calib.to(torch.float32).squeeze(0)
                img_out = visualize_objects_cv(_input, calib, tt_objects_torch, prep_config=org_prep_configs[i] if args.full_res else None)
                if args.output is not None:
                    if os.path.isdir(args.output):
                        os.makedirs(args.output, exist_ok=True)
                        basename, ext = os.path.splitext(os.path.basename(org_input_paths[i]))
                        output_path = os.path.join(args.output, f"{basename}_out{ext}")
                    else:
                        output_path = args.output
                    print(f"Saving output image to {output_path}")
                    cv2.imwrite(output_path, img_out)
                with lock:
                    shared_dict["output_image"] = img_out

            if not shared_dict["running"] or iter >= num_iter:
                shared_dict["running"] = False
                img_out = cv2.putText(img_out, "Press q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                with lock:
                    shared_dict["output_image"] = img_out
                break
        cv2_thread.join()

        pipeline.cleanup()


if __name__ == "__main__":
    args = parse_args()
    device = ttnn.CreateDevice(
        device_id=0,
        l1_small_size=32768,
        num_command_queues=2,
        trace_region_size=3000000,
    )
    from conftest import model_location_generator

    image_exts = {".jpg", ".jpeg", ".png"}
    if os.path.isdir(args.input):
        assert os.path.isdir(args.calib), "If --input is a directory, --calib must also be a directory."
        input_image_paths = []
        calib_paths = []
        calib_list = os.listdir(args.calib)
        for f in os.listdir(args.input):
            full_path = os.path.join(args.input, f)

            if os.path.isfile(full_path):
                basename, ext = os.path.splitext(f)
                ext = ext.lower()
                calib_candi_name = basename + ".txt"
                if ext in image_exts and calib_candi_name in calib_list:
                    input_image_paths.append(full_path)
                    calib_full_path = os.path.join(args.calib, calib_candi_name)
                    calib_paths.append(calib_full_path)
    else:
        input_image_paths = [args.input]
        calib_paths = [args.calib]

    main(device, model_location_generator, input_image_paths, calib_paths, args)
    ttnn.close_device(device)
