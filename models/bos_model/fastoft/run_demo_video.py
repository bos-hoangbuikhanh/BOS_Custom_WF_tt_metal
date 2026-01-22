# for display
import argparse
import multiprocessing

# import matplotlib
# matplotlib.use("QtAgg")
import cv2
import time

import os
import torch
import numpy as np
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
        "-n", "--num_videos", type=int, default=0, help="the number of videos to play. -1 is infinite loop"
    )
    parser.add_argument("-i", "--input", type=str, default=os.path.join(os.path.dirname(__file__), "resources/videos"), help="input video directory or a single video path")
    parser.add_argument("-c", "--calib", type=str, default=os.path.join(os.path.dirname(__file__), "resources/000013.txt"), help="a single calibration file path")

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

def frame_to_tensor(frame, pad_hw=(H_PADDED, W_PADDED), dtype=torch.float32):
    """Convert a BGR video frame to a preprocessed tensor."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to float and normalize to [0, 1]
    frame_float = frame_rgb.astype(np.float32) / 255.0

    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(frame_float).permute(2, 0, 1)

    # Pad to required size
    pad_h, pad_w = pad_hw
    c, h, w = tensor.shape

    if h != pad_h or w != pad_w:
        padded = torch.zeros((c, pad_h, pad_w), dtype=dtype)
        padded[:, :h, :w] = tensor
        tensor = padded

    return tensor.to(dtype)

from models.bos_model.fastoft.tests.test_perf_e2e_oft import OFTPerformanceRunnerInfra

def main(device, model_location_generator, input_video_paths, calib_path, args):
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

    # Create dummy input for parameter creation
    input_tensor = torch.zeros((1, 3, H_PADDED, W_PADDED), dtype=model_dtype)

    # Use first calib/grid for model setup (all should be compatible)
    calib = load_calib(calib_path, dtype=model_dtype)[None].to(model_dtype)
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
        device, input_tensor, calib, grid, ref_model, oft_mode, model_location_generator
    )

    ttnn.synchronize_device(device)

    with torch.no_grad():
        # _run_model_pipeline with 2cqs and trace
        _num_command_queues = 1
        _trace = True

        # Use first input for warmup and compilation
        test_infra.input_tensor_host = input_tensor

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


        num_warmup_iterations = 5
        # Run warmup iterations with first input only
        if num_warmup_iterations > 0:
            print(f"** Warmup {num_warmup_iterations} iterations **")
            warmup_inputs = [tt_inputs_host] * num_warmup_iterations
            pipeline.preallocate_output_tensors_on_host(num_warmup_iterations)
            pipeline.enqueue(warmup_inputs).pop_all()
            print(f"** Warmup {num_warmup_iterations} iterations complete **")

    total_video_count = 0
    total_frame_count = 0
    print("** Starting video processing **")
    if args.num_videos == 0:
        args.num_videos = len(input_video_paths)
    try:
        cv2_thread.start()
        while True:
            video_index = total_video_count % len(input_video_paths)
            input_video_path = input_video_paths[video_index]
            # Open video
            print(f"Opening video file: {input_video_path}")
            cap = cv2.VideoCapture(input_video_path)
            assert cap.isOpened(), f"Could not open video file {input_video_path}"

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                profiler.start("frame_processing")
                # Convert frame to tensor
                input_tensor = frame_to_tensor(frame, dtype=model_dtype)[None]
                test_infra.input_tensor_host = input_tensor
                tt_input_host, _, _ = test_infra.setup_dram_sharded_input(device)
                host_inputs = [tt_input_host]

                # Preallocate outputs
                pipeline.preallocate_output_tensors_on_host(1)

                profiler.start("model_processing")
                outputs = pipeline.enqueue(host_inputs).pop_all()
                profiler.end("model_processing")

                # Visualize predictions
                torch_outputs = test_infra.postprocess(outputs[0])
                tt_objects_torch = test_infra.tt_encoder.create_objects(*torch_outputs)
                input_tensor = input_tensor.to(torch.float32).squeeze(0)
                calib = calib.to(torch.float32).squeeze(0)
                img_out = visualize_objects_cv(input_tensor, calib, tt_objects_torch, cmap=(255, 217, 0))
                with lock:
                    shared_dict["output_image"] = img_out

                profiler.end("frame_processing")
                frame_count += 1
                total_frame_count += 1
                model_inference_time = profiler.get("model_processing")
                e2e_inference_time = profiler.get("frame_processing")
                print(f"[{total_video_count}/{total_frame_count}] ({video_index})-video-({frame_count})-frame processed model FPS: {(1/model_inference_time):.2f} Hz, E2E FPS: {(1/e2e_inference_time):.2f} Hz")

                if not shared_dict["running"]:  # stop signal received from cv2 thread
                    break  # exit video frame loop
            cap.release()
            if not shared_dict["running"]:
                break  # exit every video loop

            total_video_count += 1
            if args.num_videos >= 0 and total_video_count >= args.num_videos:
                shared_dict["running"] = False
                img_out = cv2.putText(img_out, "Press q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                with lock:
                    shared_dict["output_image"] = img_out
                break
    finally:
        # breakpoint()
        shared_dict["running"] = False
        cv2_thread.join()

if __name__ == "__main__":
    args = parse_args()
    device = ttnn.CreateDevice(
        device_id=0,
        l1_small_size=16 * 1024,
        num_command_queues=2,
        trace_region_size=3000000,
    )
    from conftest import model_location_generator

    video_exts = {".mp4", ".avi", ".mov"}
    if os.path.isdir(args.input):
        input_video_paths = []
        for f in sorted(os.listdir(args.input)):
            full_path = os.path.join(args.input, f)

            if os.path.isfile(full_path):
                if os.path.splitext(f)[1].lower() in video_exts:
                    input_video_paths.append(full_path)
    else:
        input_video_paths = [args.input]

    main(device, model_location_generator, input_video_paths, args.calib, args)
    ttnn.close_device(device)
