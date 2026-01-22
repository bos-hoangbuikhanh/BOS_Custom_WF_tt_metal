import argparse
import os

os.environ["LOGURU_LEVEL"] = "INFO"
from loguru import logger

import ttnn
from models.bos_model.demo.model_task.classification.dataset_utils import IMAGENET_LABEL_DICT, get_data_loader
from models.bos_model.demo.model_task.classification.model_utils import SUPPORTED_MODEL_IDS, get_model


def launch_app(device_id, batch_size, **kwargs):
    model_id = kwargs.get("model_id", None)
    use_trace = kwargs.get("use_trace", False)
    use_2cq = kwargs.get("use_2cq", False)
    data_path = kwargs.get("data_path", "models/bos_model/demo/dataset/sample/")
    data_shuffle = kwargs.get("data_shuffle", False)
    data_seed = kwargs.get("data_seed", 0)

    if args.model not in SUPPORTED_MODEL_IDS:
        raise NotImplementedError(f"{args.model} is not supported yet")

    device = ttnn.CreateDevice(
        device_id=device_id,
        l1_small_size=32768,
        trace_region_size=1605632 if use_trace else ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
        num_command_queues=2 if use_2cq else 1,
    )
    if kwargs.get("enable_proram_cache", False):
        device.enable_program_cache()

    model_mode = [True, False]
    model = get_model(device, model_id, batch_size, use_trace, use_2cq, model_mode)

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

    # Trace capture
    if use_trace:
        model.trace_capture()

    img_idx = 0
    for it, batch in enumerate(data_loader):
        start_idx = it * batch_size
        end_idx = min(start_idx + batch_size, num_images)

        logger.info(f"Processing batch {it+1}/{iterations}: images {start_idx+1}-{end_idx}")

        tt_output = model(batch["pixel_values"])  # More optimized run with caching
        tt_output = ttnn.from_device(tt_output, blocking=True)
        batch_predictions = ttnn.to_torch(tt_output)[:, 0, :1000].argmax(dim=-1).tolist()

        batch_images = {
            "image": dataset["image"][img_idx : img_idx + batch_size],
            "label": dataset["label"][img_idx : img_idx + batch_size],
        }

        for pr, gt in zip(batch_predictions, batch_images["label"]):
            logger.info(f"  Expected Label: {IMAGENET_LABEL_DICT[gt]}, Predicted Label: {IMAGENET_LABEL_DICT[pr]}")

        img_idx += batch_size

    # Trace release
    if use_trace:
        ttnn.release_trace(device, model.tid)
    else:
        ttnn.deallocate(tt_output, force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification GUI Demo")
    parser.add_argument("--device_id", type=int, default=0, help="device id to use for inference")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(SUPPORTED_MODEL_IDS.keys()),
        default=list(SUPPORTED_MODEL_IDS.keys())[0],
        help="model id for inference",
    )
    parser.add_argument("-b", "--batch", type=int, default=4, help="Size of batch")
    parser.add_argument("--trace", action="store_true", help="Use trace")
    parser.add_argument("--cq2", action="store_true", help="Use 2 command queue")
    parser.add_argument(
        "--data_dir", type=str, default="models/bos_model/demo/dataset/sample/", help="directory path to image data"
    )
    parser.add_argument("--no_shuffle", action="store_true", help="do not shuffle file list at loader init")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed for file list")
    args = parser.parse_args()

    cfg = {
        "enable_proram_cache": True,
        "model_id": args.model,
        "use_trace": args.trace,
        "use_2cq": args.cq2,
        "data_path": args.data_dir,
        "data_shuffle": args.no_shuffle,
        "data_seed": args.seed,
    }
    launch_app(args.device_id, args.batch, **cfg)

# python models/bos_model/demo/model_task/classification/runner.py --model google/vit-base-patch16-224 --data_dir models/bos_model/demo/dataset/imagenet-val/
# python models/bos_model/demo/model_task/classification/runner.py --model microsoft/resnet-50 --data_dir models/bos_model/demo/dataset/imagenet-val/
