import argparse
import os
import queue
import time
from threading import Event, Thread

os.environ["LOGURU_LEVEL"] = "INFO"
from loguru import logger

import ttnn
from models.bos_model.demo.model_task.classification.dataset_utils import IMAGENET_LABEL_DICT, get_data_loader
from models.bos_model.demo.model_task.classification.demo_utils import (
    compose_sidebar_result,
    draw_result_window,
    draw_sidebar,
    ui_thread,
)
from models.bos_model.demo.model_task.classification.model_utils import SUPPORTED_MODEL_IDS, get_model


def resnet_runner(device_id, batch_size=4, num_iter=1, **kwargs):
    assert batch_size in [1, 2, 4], "Only batch size 1, 2 and 4 is supported in ResNet50"

    model_id = "microsoft/resnet-50"
    model_name = SUPPORTED_MODEL_IDS[model_id]["model_name"]
    use_trace = kwargs.get("use_trace", False)
    use_2cq = kwargs.get("use_2cq", False)
    data_path = kwargs.get("data_path", "models/bos_model/demo/dataset/sample/")
    data_shuffle = kwargs.get("data_shuffle", False)
    data_seed = kwargs.get("data_seed", 0)
    delay_time_sec = kwargs.get("delay_time_sec", 0.0)
    demo_vis = kwargs.get("demo_vis", False)
    demo_fullscreen = kwargs.get("demo_fullscreen", False)
    demo_window_name = kwargs.get("demo_window_name", "[BOS] Image Classification Demo")
    benchmark = kwargs.get("benchmark", True)

    device = ttnn.CreateDevice(
        device_id=device_id,
        l1_small_size=32768,
        trace_region_size=1605632 if use_trace else ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
        num_command_queues=2 if use_2cq else 1,
    )
    if kwargs.get("enable_proram_cache", False):
        device.enable_program_cache()

    model_mode = [False, True] if demo_vis or benchmark else [True, False]
    model = get_model(device, model_id, batch_size, use_trace, use_2cq, model_mode)
    num_measurement_iterations = 10
    (demo_vis or benchmark) and model.set_num_measurement_iterations(num_measurement_iterations)

    data_loader = get_data_loader(data_path)(
        input_loc=data_path,
        model_id=model_id,
        batch_size=batch_size,
        shuffle=not data_shuffle,
        seed=data_seed,
    )
    dataset = {"image": data_loader.files, "label": [data_loader._label_from_path(f) for f in data_loader.files]}

    num_images = data_loader.total_files
    iterations = (num_images + batch_size - 1) // batch_size

    all_images_processed = 0
    total_inference_time = 0.0
    model_output = []

    # Trace capture
    if use_trace:
        model.trace_capture()

    if demo_vis:
        frame_queue = queue.Queue(maxsize=3)
        stop_flag = Event()
        paused_flag = Event()
        ui_t = Thread(
            target=ui_thread, args=(frame_queue, stop_flag, paused_flag, demo_window_name, demo_fullscreen), daemon=True
        )
        ui_t.start()

    for nt in range(num_iter):
        if demo_vis:
            if num_iter > 1:
                model_output = []  # for memory saving in multiple iteration runs
            if stop_flag.is_set():
                break
        img_idx = 0
        for it, batch in enumerate(data_loader):
            if demo_vis:
                if stop_flag.is_set():
                    break
                while paused_flag.is_set() and not stop_flag.is_set():
                    time.sleep(0.05)

            start_idx = it * batch_size
            end_idx = min(start_idx + batch_size, num_images)

            logger.info(
                f"Processing batch {iterations*nt+it+1}/{iterations*num_iter}: images {num_images*nt+start_idx+1}-{num_images*nt+end_idx}"
            )

            tt_output = model(batch["pixel_values"])
            batch_predictions = ttnn.to_torch(tt_output)[:, 0, :1000].argmax(dim=-1).tolist()
            model_output.extend(batch_predictions)
            all_images_processed += batch_size

            # FPS
            if demo_vis or benchmark:
                batch_inference_time = model.profiler.get("inference_batch") / num_measurement_iterations
                total_inference_time += batch_inference_time
                batch_fps = batch_size / max(1e-9, batch_inference_time)
                logger.info(
                    f"  Batch {iterations*nt+it+1} inference time: {batch_inference_time:.4f}s, FPS: {batch_fps:.2f}"
                )

            batch_images = {
                "image": dataset["image"][img_idx : img_idx + batch_size],
                "label": dataset["label"][img_idx : img_idx + batch_size],
            }

            if demo_vis:
                frame = compose_sidebar_result(
                    draw_sidebar(model_name, batch_size, batch_fps),
                    draw_result_window(batch_size, batch_images, batch_predictions, IMAGENET_LABEL_DICT),
                )
                if not stop_flag.is_set():
                    frame_queue.put(frame)

            if delay_time_sec != 0 and not paused_flag.is_set() and not stop_flag.is_set():
                time.sleep(delay_time_sec)

            for pr, gt in zip(batch_predictions, batch_images["label"]):
                logger.info(
                    f"      Expected Label: {IMAGENET_LABEL_DICT[gt]}, Predicted Label: {IMAGENET_LABEL_DICT[pr]}"
                )

            img_idx += batch_size

    if demo_vis:
        frame_queue.put(None)
        stop_flag.set()
        ui_t.join()

    # Trace release
    if use_trace:
        ttnn.release_trace(device, model.tid)
    else:
        ttnn.deallocate(tt_output, force=True)

    # performance metric
    if demo_vis or benchmark:
        avg_inference_time = total_inference_time / ((all_images_processed // batch_size) * num_iter)
        total_fps = all_images_processed / max(1e-12, total_inference_time)

        warmup_time = model.profiler.get("warmup")

        logger.info("=" * 50)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total images processed               : {all_images_processed}")
        logger.info(f"Batch size                           : {batch_size*device.get_num_devices()}")
        logger.info(f"Warmup time                          : {warmup_time:.4f}s")
        logger.info(f"Total inference time                 : {total_inference_time:.4f}s")
        logger.info(f"Average inference time per batch     : {avg_inference_time:.4f}s")
        logger.info(f"Overall FPS (total images/total time): {total_fps:.4f}")
        logger.info("=" * 50)

        return {
            "fps": total_fps,
            "output_tensor": model_output,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification GUI Demo")

    # for model
    parser.add_argument("--device_id", type=int, default=0, help="device id to use for inference")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Size of batch")
    parser.add_argument("--trace", action="store_true", help="Use trace")
    parser.add_argument("--cq2", action="store_true", help="Use 2 command queue")
    parser.add_argument(
        "--data_dir", type=str, default="models/bos_model/demo/dataset/sample/", help="directory path to image data"
    )
    parser.add_argument("--no_shuffle", action="store_true", help="do not shuffle file list at loader init")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed for file list")

    # for demo
    parser.add_argument("-delay", type=float, default=0.0, help="Delay time(sec), floating number")
    parser.add_argument("-n", "--num_iters", type=int, default=1, help="Number of iteration")
    parser.add_argument("--demo", action="store_true", help="Show demo window")
    parser.add_argument("--fullscreen", action="store_true", help="Show demo window full size screen")

    # for benchmark
    parser.add_argument("--benchmark", action="store_true", help="Enable profiling for benchmark")

    args = parser.parse_args()

    cfg = {
        "enable_proram_cache": True,
        "use_trace": args.trace,
        "use_2cq": args.cq2,
        "data_path": args.data_dir,
        "data_shuffle": args.no_shuffle,
        "data_seed": args.seed,
        "delay_time_sec": args.delay,
        "demo_vis": args.demo,
        "demo_fullscreen": args.fullscreen,
        "demo_window_name": "[BOS] Image Classification Demo",
        "benchmark": args.benchmark,
    }
    resnet_runner(args.device_id, args.batch_size, args.num_iters, **cfg)
