# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.bos_model.yolo.yolov10s.common import YOLOV10_L1_SMALL_SIZE, load_torch_model
from models.bos_model.yolo.yolov10s.demo.demo_utils import postprocess
from models.bos_model.yolo.yolov10s.reference.yolov10s import YOLOv10
from models.bos_model.yolo.yolov10s.runner.performant_runner_infra import YOLOv10PerformanceRunnerInfra
from models.bos_model.yolo.yolov10s.runner.pipeline_runner import YoloV10sPipelineRunner
from models.common.utility_functions import disable_persistent_kernel_cache
from models.demos.utils.common_demo_utils import (
    LoadImages,
    get_mesh_mappers,
    load_coco_class_names,
    preprocess,
    save_yolo_predictions_by_model,
)
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


def init_model_and_runner(
    model_location_generator, device, model_type, use_weights_from_ultralytics, batch_size_per_device, res=(320, 320)
):
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")

    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    if use_weights_from_ultralytics:
        torch_model = load_torch_model(model_location_generator)
        state_dict = torch_model.state_dict()

    torch_model = YOLOv10()
    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    model = torch_model

    logger.info("Inferencing [Torch] Model")

    test_infra = None
    if model_type == "tt_model":
        test_infra = YOLOv10PerformanceRunnerInfra(
            device,
            batch_size_per_device,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            model_location_generator,
            resolution=res,
            use_pretrained_weight=True,
            mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
            mesh_composer=outputs_mesh_composer,
        )

    return model, test_infra, outputs_mesh_composer, batch_size


def process_images(dataset, res, batch_size):
    torch_images, orig_images, paths_images = [], [], []

    for paths, im0s, _ in dataset:
        assert len(im0s) == batch_size, f"Expected batch of size {batch_size}, but got {len(im0s)}"

        paths_images.extend(paths)
        orig_images.extend(im0s)

        for idx, img in enumerate(im0s):
            if img is None:
                raise ValueError(f"Could not read image: {paths[idx]}")
            tensor = preprocess([img], res=res)
            torch_images.append(tensor)

        if len(torch_images) >= batch_size:
            break

    torch_input_tensor = torch.cat(torch_images, dim=0)
    return torch_input_tensor, orig_images, paths_images


def run_inference_and_save(
    model, test_infra, model_type, outputs_mesh_composer, im_tensor, orig_images, paths_images, save_dir, names, device
):
    if model_type == "torch_model":
        preds = model(im_tensor)
    else:
        # Get memory configs from the infrastructure
        tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)

        config = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
        pipeline = create_pipeline_from_config(
            config,
            YoloV10sPipelineRunner(test_infra),
            device,
            dram_input_memory_config=sharded_mem_config_DRAM,
            l1_input_memory_config=input_mem_config,
        )

        # Compile pipeline
        pipeline.compile(tt_inputs_host)

        # Convert input to TTNN format using the infrastructure's method
        tt_inputs_host, _ = test_infra._setup_l1_sharded_input(device, im_tensor)

        # Run inference
        outputs = pipeline.enqueue([tt_inputs_host]).pop_all()
        preds = outputs[0]

        preds = ttnn.to_torch(preds, dtype=torch.float32, mesh_composer=outputs_mesh_composer)
        pipeline.cleanup()

    results = postprocess(preds, im_tensor, orig_images, paths_images, names)

    for result, image_path in zip(results, paths_images):
        save_yolo_predictions_by_model(result, save_dir, image_path, model_type)


def run_yolov10s_demo(
    model_location_generator, device, model_type, use_weights_from_ultralytics, res, input_loc, batch_size_per_device
):
    model, test_infra, mesh_composer, batch_size = init_model_and_runner(
        model_location_generator, device, model_type, use_weights_from_ultralytics, batch_size_per_device, res=res
    )

    dataset = LoadImages(path=os.path.abspath(input_loc), batch=batch_size)
    im_tensor, orig_images, paths_images = process_images(dataset, res, batch_size)
    names = load_coco_class_names()
    save_dir = "models/bos_model/yolo/yolov10s/demo/outputs"

    run_inference_and_save(
        model, test_infra, model_type, mesh_composer, im_tensor, orig_images, paths_images, save_dir, names, device
    )

    logger.info("Inference done")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV10_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model",  # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize(
    "res",
    [
        (224, 224),
        (320, 320),
    ],
    ids=["res224", "res320"],
)
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device ",
    [
        (
            "models/bos_model/yolo/yolov10s/demo/images/",
            1,
        ),
    ],
)
def test_demo(
    model_location_generator, device, model_type, use_weights_from_ultralytics, res, input_loc, batch_size_per_device
):
    run_yolov10s_demo(
        model_location_generator,
        device,
        model_type,
        use_weights_from_ultralytics,
        res,
        input_loc,
        batch_size_per_device,
    )
