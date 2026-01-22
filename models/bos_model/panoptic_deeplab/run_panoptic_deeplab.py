import argparse
import os
from dataclasses import dataclass
from typing import Callable, Tuple

import cv2
import numpy as np
import pytest
import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image
from torch.nn import functional as F

import ttnn
from models.bos_model.panoptic_deeplab.demo.demo_utils import (
    create_deeplab_v3plus_visualization,
    create_panoptic_visualization,
    preprocess_image,
    preprocess_input_params,
    save_predictions,
)
from models.bos_model.panoptic_deeplab.reference.pytorch_model import (
    DEEPLAB_V3_PLUS,
    PANOPTIC_DEEPLAB,
    PytorchPanopticDeepLab,
)
from models.bos_model.panoptic_deeplab.tt.common import (
    PDL_L1_SMALL_SIZE,
    get_panoptic_deeplab_config,
    get_panoptic_deeplab_weights_path,
    preprocess_nchw_input_tensor,
)
from models.bos_model.panoptic_deeplab.tt.model_configs import ModelOptimisations
from models.bos_model.panoptic_deeplab.tt.model_preprocessing import (
    create_panoptic_deeplab_parameters,
    fuse_conv_bn_parameters,
)
from models.bos_model.panoptic_deeplab.tt.tt_custom_pipeline import (
    CustomTracedModelExecutor,
    create_pipeline_from_config,
)
from models.bos_model.panoptic_deeplab.tt.tt_model import TtPanopticDeepLab
from models.common.utility_functions import profiler
from models.tt_cnn.tt.executor import ModelExecutor
from models.tt_cnn.tt.pipeline import PipelineConfig

# from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from tests.ttnn.utils_for_testing import check_with_pcc

# from models.bos_model.panoptic_deeplab.utilities.norm_utils import BitMasks, ImageList, Instances
# from models.bos_model.panoptic_deeplab.utilities.post_processing import (
#     ResizeShortestEdge,
#     get_panoptic_segmentation,
#     sem_seg_postprocess,
# )
# from detectron2.data import MetadataCatalog

pdl_home_path = os.path.dirname(os.path.realpath(__file__))
IMAGES_PATH = os.path.join(pdl_home_path, "images")
WEIGHTS_PATH = os.path.join(pdl_home_path, "weights", "model_final_bd324a.pkl")
OUTPUT_PATH = os.path.join(pdl_home_path, "output")
center_threshold = 0.05


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("images", nargs="?", type=str, default=IMAGES_PATH, help="Images to process")
    parser.add_argument(
        "weights_path", nargs="?", type=str, default=WEIGHTS_PATH, help="Location of model weights file"
    )
    parser.add_argument("output_path", nargs="?", type=str, default=OUTPUT_PATH, help="Location to store model outputs")

    parser.add_argument("--image_height", type=int, default=256, help="Height of output processed by bos model")
    parser.add_argument("--image_width", type=int, default=512, help="Eidth of output processed by bos model")
    parser.add_argument("--trace", action="store_true", help="Enable Trace mode")
    parser.add_argument("-p", "--persistent_cache", action="store_true", help="Enable Persistent Cache mode")
    parser.add_argument("--enable_logger", action="store_true", help="Enable Logger outputs")
    parser.add_argument(
        "--model_category",
        type=str,
        default="DEEPLAB_V3_PLUS",
        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"],
        help="Model category: DEEPLAB_V3_PLUS (semantic segmentation only, faster) or PANOPTIC_DEEPLAB (full model with instance segmentation)",
    )

    args, _ = parser.parse_known_args(argv)
    return args


@dataclass
class ExecutorTestConfig:
    """Configuration for testing different executor types."""

    name: str
    use_trace: bool
    num_command_queues: int
    all_transfers_on_separate_command_queue: bool = False
    requires_output_config: bool = False
    requires_minimum_inputs: int = 1
    expected_executor_type: type = None


def create_host_input_tensors(
    device: ttnn.Device, batch_size: int, input_height: int, input_width: int, input_images: list
) -> Tuple[list, ttnn.MemoryConfig, ttnn.MemoryConfig]:
    """
    Create host input tensors for Panoptic DeepLab.

    Uses interleaved DRAM to avoid core grid constraints (especially for traced executors),
    and converts to L1 using to_memory_config which handles the interleaved-to-sharded conversion.

    Args:
        device: TTNN device
        batch_size: Batch size (should be 1 for Panoptic DeepLab)
        input_height: Input image height
        input_width: Input image width
        num_inputs: Number of input tensors to create

    Returns:
        Tuple of (list of host input tensors, dram_memory_config, l1_memory_config)
        - dram_memory_config: Interleaved DRAM (no core constraints)
        - l1_memory_config: Sharded L1 with full grid (original sharding from preprocessed tensor)
    """
    original_images = []
    host_inputs = []
    dram_memory_config = None
    l1_memory_config = None

    transform = T.Compose(
        [
            # T.Resize([input_height, input_width]),
            T.ToTensor(),
        ]
    )

    for i, image_file in enumerate(input_images):
        # torch_input = torch.randn(batch_size, 3, input_height, input_width, dtype=torch.bfloat16)
        # Load image from file
        # image = Image.open(image_file).convert("RGB")
        # torch_input = transform(image)
        original_image = cv2.imread(image_file)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(original_image, (input_width, input_height))
        original_images.append(original_image)
        torch_input = transform(Image.fromarray(original_image))
        torch_input = torch_input.unsqueeze(0).to(dtype=torch.bfloat16)
        # Preprocess to TTNN format (height-sharded on device)
        ttnn_input = preprocess_nchw_input_tensor(device, torch_input)

        # Extract memory config from first tensor
        if i == 0:
            # Get the L1 memory config from the preprocessed tensor (full grid sharding)
            original_mem_config = ttnn_input.memory_config()
            original_shard_spec = original_mem_config.shard_spec

            # Always use interleaved DRAM (no core constraints)
            # This avoids the logical grid constraint issue with traced executors
            # The executor will use to_memory_config to convert from interleaved DRAM to sharded L1
            dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

            # Use the original L1 sharding (full grid) - to_memory_config will handle conversion
            l1_memory_config = ttnn.MemoryConfig(
                original_mem_config.memory_layout,
                ttnn.BufferType.L1,
                original_shard_spec,
            )

        # Convert to host tensor for pipeline
        host_input = ttnn_input.cpu()
        host_inputs.append(host_input)

    return original_images, host_inputs, dram_memory_config, l1_memory_config


def create_model_wrapper(ttnn_model: TtPanopticDeepLab) -> Callable:
    """
    Create a model wrapper function for use with pipeline/executor.

    The wrapper takes an L1 input tensor and calls the model's forward method.
    The executor expects the model to accept L1 tensors and return device tensors.

    Args:
        ttnn_model: The TtPanopticDeepLab model instance

    Returns:
        A callable function that takes an L1 input tensor and returns model outputs
    """

    def model_forward(l1_input_tensor: ttnn.Tensor):
        """
        Forward pass wrapper for pipeline executor.

        Args:
            l1_input_tensor: Input tensor in L1 memory (expected by executor)

        Returns:
            Tuple of (semantic_logits, center_heatmap, offset_map)
            For DEEPLAB_V3_PLUS, returns only semantic_logits (not a tuple with None)
        """
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"

        # Call model forward
        semantic_logits, center_heatmap, offset_map, _ = ttnn_model.forward(l1_input_tensor, return_features=False)

        # For DEEPLAB_V3_PLUS, center_heatmap and offset_map are None
        # Return only semantic_logits to avoid None handling issues in executor
        if ttnn_model.model_category == DEEPLAB_V3_PLUS:
            return semantic_logits
        else:
            # Return as tuple for PANOPTIC_DEEPLAB
            return (semantic_logits, center_heatmap, offset_map)

    return model_forward


def preprocessing(cfg, batched_inputs):
    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)

    images = [x["image"] for x in batched_inputs]
    images = [(x - pixel_mean) / pixel_std for x in images]
    # To avoid error in ASPP layer when input has different size.
    size_divisibility = 0
    images = ImageList.from_tensors(images, size_divisibility)
    return images


def postprocessing(cfg, batched_inputs, outputs, images):
    # Permute TTNN outputs back to torch shape
    # sem_seg_results = ttnn.to_torch(outputs["sem_seg_results"])
    # # ttnn.deallocate(outputs["sem_seg_results"])
    # center_results = ttnn.to_torch(outputs["center_results"])
    # # ttnn.deallocate(outputs["center_results"])
    # offset_results = ttnn.to_torch(outputs["offset_results"])
    # # ttnn.deallocate(outputs["offset_results"])

    sem_seg_results = outputs["sem_seg_results"]
    center_results = outputs["center_results"]
    offset_results = outputs["offset_results"]

    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    stuff_area = cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA
    threshold = cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD
    nms_kernel = cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL
    top_k = cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE
    predict_instances = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES
    assert (
        cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV == cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
    )

    processed_results = []
    for sem_seg_result, center_result, offset_result, input_per_image, image_size in zip(
        sem_seg_results, center_results, offset_results, batched_inputs, images.image_sizes
    ):
        height = input_per_image.get("height")
        width = input_per_image.get("width")
        r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
        c = sem_seg_postprocess(center_result, image_size, height, width)
        o = sem_seg_postprocess(offset_result, image_size, height, width)
        # Post-processing to get panoptic segmentation.
        panoptic_image, _ = get_panoptic_segmentation(
            r.argmax(dim=0, keepdim=True),
            c,
            o,
            thing_ids=meta.thing_dataset_id_to_contiguous_id.values(),
            label_divisor=meta.label_divisor,
            stuff_area=stuff_area,
            void_label=-1,
            threshold=threshold,
            nms_kernel=nms_kernel,
            top_k=top_k,
        )
        # For semantic segmentation evaluation.
        processed_results.append({"sem_seg": r})
        panoptic_image = panoptic_image.squeeze(0)
        semantic_prob = F.softmax(r, dim=0)
        # For panoptic segmentation evaluation.
        processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
        # For instance segmentation evaluation.
        if predict_instances:
            instances = []
            panoptic_image_cpu = panoptic_image.cpu().numpy()
            for panoptic_label in np.unique(panoptic_image_cpu):
                if panoptic_label == -1:
                    continue
                pred_class = panoptic_label // meta.label_divisor
                isthing = pred_class in list(meta.thing_dataset_id_to_contiguous_id.values())
                # Get instance segmentation results.
                if isthing:
                    instance = Instances((height, width))
                    # Evaluation code takes continuous id starting from 0
                    instance.pred_classes = torch.tensor([pred_class], device=panoptic_image.device)
                    mask = panoptic_image == panoptic_label
                    instance.pred_masks = mask.unsqueeze(0)
                    # Average semantic probability
                    sem_scores = semantic_prob[pred_class, ...]
                    sem_scores = torch.mean(sem_scores[mask])
                    # Center point probability
                    mask_indices = torch.nonzero(mask).float()
                    center_y, center_x = (
                        torch.mean(mask_indices[:, 0]),
                        torch.mean(mask_indices[:, 1]),
                    )
                    center_scores = c[0, int(center_y.item()), int(center_x.item())]
                    # Confidence score is semantic prob * center prob.
                    instance.scores = torch.tensor([sem_scores * center_scores], device=panoptic_image.device)
                    # Get bounding boxes
                    instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
                    instances.append(instance)
            if len(instances) > 0:
                processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results


def test_panoptic_deeplab_inference(device, **kwargs):
    config = get_panoptic_deeplab_config()
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]
    input_height, input_width = config["train_size"]

    try:
        pytorch_model = PytorchPanopticDeepLab(
            num_classes=num_classes,
            common_stride=config["common_stride"],
            project_channels=config["project_channels"],
            decoder_channels=config["decoder_channels"],
            sem_seg_head_channels=config["sem_seg_head_channels"],
            ins_embed_head_channels=config["ins_embed_head_channels"],
            train_size=config["train_size"],
            weights_path=kwargs["weights_path"],
            model_category=kwargs["model_category"],
        )
        pytorch_model = pytorch_model.to(dtype=torch.bfloat16)
        pytorch_model.eval()

    except FileNotFoundError:
        pytest.fail("model_final_bd324a.pkl file not found. Please place the weights file in the weights folder.")

    # Create TTNN parameters
    ttnn_parameters = create_panoptic_deeplab_parameters(
        pytorch_model, device, input_height=input_height, input_width=input_width, batch_size=batch_size
    )

    # Apply Conv+BatchNorm fusion
    fused_parameters = fuse_conv_bn_parameters(ttnn_parameters, eps=1e-5)

    # Create model configurations
    model_configs = ModelOptimisations(
        conv_act_dtype=ttnn.bfloat8_b,
        conv_w_dtype=ttnn.bfloat8_b,
    )
    model_configs.setup_resnet_backbone()
    model_configs.setup_aspp()
    model_configs.setup_decoder()
    model_configs.setup_heads()

    # Create TTNN model
    ttnn_model = TtPanopticDeepLab(
        device=device,
        parameters=fused_parameters,
        num_classes=num_classes,
        common_stride=config["common_stride"],
        project_channels=config["project_channels"],
        decoder_channels=config["decoder_channels"],
        sem_seg_head_channels=config["sem_seg_head_channels"],
        ins_embed_head_channels=config["ins_embed_head_channels"],
        train_size=config["train_size"],
        model_configs=model_configs,
        model_category=kwargs["model_category"],
    )

    # Create model wrapper for pipeline
    model_wrapper = create_model_wrapper(ttnn_model)

    # Create host input tensors
    original_images, host_inputs, dram_memory_config, l1_memory_config = create_host_input_tensors(
        device, batch_size, input_height, input_width, kwargs["images"]
    )
    num_inputs = len(host_inputs)

    # Create pipeline using extracted memory configs
    pipeline_config = PipelineConfig(
        use_trace=kwargs["executor_config"].use_trace,
        num_command_queues=kwargs["executor_config"].num_command_queues,
        all_transfers_on_separate_command_queue=kwargs["executor_config"].all_transfers_on_separate_command_queue,
    )

    pipeline_args = {
        "config": pipeline_config,
        "model": model_wrapper,
        "device": device,
        "dram_input_memory_config": dram_memory_config,
        "l1_input_memory_config": l1_memory_config,
    }

    pipe = create_pipeline_from_config(**pipeline_args)

    # Verify executor type
    assert isinstance(
        pipe.executor, executor_config.expected_executor_type
    ), f"Expected {executor_config.expected_executor_type.__name__}, got {type(pipe.executor).__name__}"

    # Compile pipeline
    logger.info(f"Compiling pipeline with {executor_config.name}...")
    pipe.compile(host_inputs[0])

    timing_key = f"pipeline_execution_{executor_config.name}_{kwargs['model_category']}"
    profiler.clear()
    profiler.enable()
    profiler.start(timing_key)
    outputs = pipe.enqueue(host_inputs).pop_all()
    profiler.end(timing_key, PERF_CNT=num_inputs)

    # for out in outputs:
    #     for tensor in out:
    #         print(tensor.shape, tensor.memory_config())

    # Store timing results for later printing (after PCC)
    avg_execution_time = profiler.get(timing_key)
    total_execution_time = avg_execution_time * num_inputs
    avg_execution_time_us = avg_execution_time * 1e6  # Convert to microseconds
    samples_per_second = 1.0 / avg_execution_time if avg_execution_time > 0 else 0

    print("================ Execution Results ================")
    print(f"Model Category: {kwargs['model_category']}")
    print(f"Number of Inputs: {num_inputs}")
    print(f"Average Execution Time per Input: {avg_execution_time_us:.2f} µs")
    print(f"Total Execution Time: {total_execution_time:.4f} s")
    print(f"Samples per Second: {samples_per_second:.2f} samples/s")
    print("===================================================")

    # # Generate reference outputs from PyTorch (after pipeline execution)
    # logger.info("Generating reference outputs from PyTorch model...")
    # reference_outputs = []
    # pytorch_model = pytorch_model.to(dtype=torch.float32)
    # for host_input in host_inputs:
    #     # Convert host input back to torch for reference
    #     torch_input = ttnn.to_torch(host_input)
    #     # Input is in NHWC format [1, H, W, C] where C=8 (3 original + 5 padding)
    #     # We need to remove padding and convert to NCHW for PyTorch model
    #     assert torch_input.shape[0] == 1, f"Expected batch size 1, got {torch_input.shape[0]}"
    #     # Remove padding: take only first 3 channels
    #     torch_input = torch_input[:, :, :, :3]  # [1, H, W, 3]
    #     # Convert NHWC -> NCHW
    #     torch_input = torch_input.permute(0, 3, 1, 2)  # [1, 3, H, W]

    #     with torch.no_grad():
    #         pytorch_semantic, pytorch_center, pytorch_offset, _ = pytorch_model.forward(
    #             torch_input.to(torch.float32)
    #         )
    #     reference_outputs.append((pytorch_semantic, pytorch_center, pytorch_offset))

    semantic_original_channels = ttnn_model.semantic_head.get_output_channels_for_slicing()
    center_original_channels = ttnn_model.instance_head.get_center_output_channels_for_slicing()
    offset_original_channels = ttnn_model.instance_head.get_offset_output_channels_for_slicing()

    for i, (original_image, ttnn_output) in enumerate(zip(original_images, outputs)):
        # Handle different output formats based on model category
        # Note: Pipeline converts tuple outputs to lists, so we check for both
        if kwargs["model_category"] == DEEPLAB_V3_PLUS:
            # For DEEPLAB_V3_PLUS, output is a single tensor (semantic_logits only)
            ttnn_semantic = ttnn_output
            assert isinstance(ttnn_semantic, ttnn.Tensor), f"Semantic output {i} should be ttnn.Tensor"
            assert ttnn_semantic.storage_type() == ttnn.StorageType.HOST, f"Semantic output {i} should be on host"

            # Convert to torch for visualization
            ttnn_semantic_torch = ttnn.to_torch(ttnn_semantic)
        else:
            # For PANOPTIC_DEEPLAB, output is a tuple/list of 3 tensors
            # Pipeline converts tuples to lists, so we accept both
            assert isinstance(
                ttnn_output, (tuple, list)
            ), f"Output {i} should be a tuple or list for PANOPTIC_DEEPLAB, got {type(ttnn_output)}"
            assert len(ttnn_output) == 3, f"Output {i} should have 3 elements, got {len(ttnn_output)}"
            ttnn_semantic, ttnn_center, ttnn_offset = ttnn_output

            # Validate output structure
            assert isinstance(ttnn_semantic, ttnn.Tensor), f"Semantic output {i} should be ttnn.Tensor"
            assert ttnn_semantic.storage_type() == ttnn.StorageType.HOST, f"Semantic output {i} should be on host"
            assert isinstance(ttnn_center, ttnn.Tensor), f"Center output {i} should be ttnn.Tensor"
            assert ttnn_center.storage_type() == ttnn.StorageType.HOST, f"Center output {i} should be on host"
            assert isinstance(ttnn_offset, ttnn.Tensor), f"Offset output {i} should be ttnn.Tensor"
            assert ttnn_offset.storage_type() == ttnn.StorageType.HOST, f"Offset output {i} should be on host"

            ttnn_semantic_torch = ttnn.to_torch(ttnn_semantic)
            ttnn_center_torch = ttnn.to_torch(ttnn_center)
            ttnn_offset_torch = ttnn.to_torch(ttnn_offset)

        # Convert to numpy in HWC format for visualization
        semantic_np_ttnn = (
            ttnn_semantic_torch[:, :semantic_original_channels, :, :].float().squeeze(0).permute(1, 2, 0).numpy()
        )
        center_np_ttnn = (
            ttnn_center_torch[:, :center_original_channels, :, :].float().squeeze(0).permute(1, 2, 0).numpy()
            if ttnn_model.model_category == PANOPTIC_DEEPLAB
            else None
        )
        offset_np_ttnn = (
            ttnn_offset_torch[:, :offset_original_channels, :, :].float().squeeze(0).permute(1, 2, 0).numpy()
            if ttnn_model.model_category == PANOPTIC_DEEPLAB
            else None
        )

        if ttnn_model.model_category == PANOPTIC_DEEPLAB:
            panoptic_vis_ttnn, panoptic_info_ttnn = create_panoptic_visualization(
                semantic_np_ttnn,
                center_np_ttnn,
                offset_np_ttnn,
                original_image,
                center_threshold=center_threshold,  # Use parameter
                score_threshold=center_threshold,  # Use same value for consistency
                stuff_area=1,  # Match PyTorch defaults
                top_k=1000,  # Match PyTorch defaults
                nms_kernel=11,  # Match PyTorch defaults
            )
        else:
            panoptic_vis_ttnn, panoptic_info_ttnn = create_deeplab_v3plus_visualization(
                semantic_np_ttnn,
                original_image=original_image,
            )

        # Save TTNN results
        image_name = os.path.basename(kwargs["images"][i])
        ttnn_output_dir = os.path.join(kwargs["output_path"], f"{image_name.split('.')[0]}_output")
        save_predictions(ttnn_output_dir, image_name, original_image, panoptic_vis_ttnn)


if __name__ == "__main__":
    l1_small_size = 1024 * 4
    args = parse_args()
    if not args.enable_logger:
        logger.remove()
    if args.trace:
        device = ttnn.open_device(device_id=0, l1_small_size=l1_small_size, trace_region_size=104192000)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=l1_small_size)
    device.enable_program_cache()
    if args.persistent_cache:
        ttnn.device.EnablePersistentKernelCache()

    image_extensions = {".png", ".jpg", ".jpeg"}
    images = []
    if not os.path.isdir(args.images):
        images = [args.images]
    else:
        for file in os.listdir(args.images):
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                images.append(os.path.join(args.images, file))

    executor_type = CustomTracedModelExecutor if args.trace else ModelExecutor
    executor_config = ExecutorTestConfig(
        name="ModelExecutor",
        use_trace=args.trace,
        num_command_queues=1,
        all_transfers_on_separate_command_queue=False,
        requires_minimum_inputs=1,
        expected_executor_type=executor_type,
    )

    # Convert string argument to model category constant
    model_category = PANOPTIC_DEEPLAB if args.model_category == "PANOPTIC_DEEPLAB" else DEEPLAB_V3_PLUS

    kwargs = {
        "model_category": model_category,
        "executor_config": executor_config,
        "images": images,
        "weights_path": args.weights_path,
        "output_path": args.output_path,
        "enable_persistent_cache": args.persistent_cache,
    }
    test_panoptic_deeplab_inference(device, **kwargs)
    ttnn.close_device(device)
