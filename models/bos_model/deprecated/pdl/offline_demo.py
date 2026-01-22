import argparse
import math
import os
import time

import cv2
import numpy as np
import torch

# The below dependencies are preprocessing and post processing
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.utils.visualizer import ColorMode, Visualizer
from torch.nn import functional as F

import ttnn
from models.bos_model.pdl.reference.modelling_utils import DefaultPredictor
from models.bos_model.pdl.reference.norm_utils import BitMasks, ImageList, Instances
from models.bos_model.pdl.reference.panoptic_seg import PanopticDeepLab
from models.bos_model.pdl.reference.post_processing import (
    ResizeShortestEdge,
    get_panoptic_segmentation,
    sem_seg_postprocess,
)
from models.bos_model.pdl.tt.model_processing import create_pdl_model_parameters
from models.bos_model.pdl.tt.ttnn_panoptic_seg import TtPanopticDeepLab

curr_file_path = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = "reference/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml"
CONFIG_FILE = os.path.join(curr_file_path, CONFIG_FILE)
WEIGHTS_PATH = "reference/pdl_weights.pt"
WEIGHTS_PATH = os.path.join(curr_file_path, WEIGHTS_PATH)
OUTPUT_DIR = "output"
OUTPUT_DIR = os.path.join(curr_file_path, OUTPUT_DIR)

BUFFER_SIZE = 5


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_height", type=int, default=256, help="height of output processed by bos model")
    parser.add_argument("--image_width", type=int, default=512, help="width of output processed by bos model")
    parser.add_argument(
        "--source",
        type=str,
        default=os.path.join(curr_file_path, "videos/car.mp4"),
        help="Video source: webcam index (e.g., 0) or path/URL",
    )
    parser.add_argument("-f", "--frames", type=int, default=0, help="Number of frames executed")
    parser.add_argument("--torch", action="store_true", help="Use Torch model")

    args = parser.parse_args()
    return args


def preprocessing(cfg, image, aug, target_height=256, target_width=512):
    original_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    if cfg.INPUT.FORMAT == "RGB":
        original_image = original_image[:, :, ::-1]
    height, width = original_image.shape[:2]
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    image = image.to(cfg.MODEL.DEVICE)

    image_dic = [{"image": image, "height": height, "width": width}]

    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1).to(cfg.MODEL.DEVICE)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1).to(cfg.MODEL.DEVICE)

    images = [x["image"] for x in image_dic]
    images = [(x - pixel_mean) / pixel_std for x in images]
    # To avoid error in ASPP layer when input has different size.
    size_divisibility = 0
    images = ImageList.from_tensors(images, size_divisibility)
    return image_dic, images


def postprocessing(cfg, batched_inputs, outputs, images, original_h=256, original_w=512):
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

        panoptic_seg_tensor, segments_info = processed_results[0]["panoptic_seg"]
        # Resize segmentation map back to original size
        panoptic_seg_np = panoptic_seg_tensor.to("cpu").numpy()
        panoptic_seg_resized = cv2.resize(
            panoptic_seg_np.astype(np.float32), (original_w, original_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

        panoptic_seg_for_visualizer = torch.from_numpy(panoptic_seg_resized)

        # return panoptic_seg_for_visualizer, segments_info

        sem_cls = r.argmax(dim=0).to("cpu").numpy().astype(np.int32)
        sem_cls_resized = cv2.resize(
            sem_cls.astype(np.float32), (original_w, original_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

        return panoptic_seg_for_visualizer, sem_cls_resized


def preprate_ttnn_inputs(device, input_tensor):
    in_n, in_c, in_h, in_w = input_tensor.shape
    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
    ttnn_input = ttnn.reshape(ttnn_input, (1, 1, in_n * in_h * in_w, in_c))

    return ttnn_input


def setup_l1_sharded_input(device, torch_input_tensor=None, min_channels=3, num_cores=20):
    if num_cores == 20:
        core_grid = ttnn.CoreGrid(y=4, x=5)
    elif num_cores == 16:
        core_grid = ttnn.CoreGrid(y=4, x=4)
    else:
        core_grid = ttnn.CoreGrid(y=8, x=8)
    torch_input_tensor = torch_input_tensor.permute((0, 2, 3, 1))  # NCHW -> NHWC
    n, h, w, c = torch_input_tensor.shape
    if c < min_channels:
        channel_padding_needed = min_channels - c
        torch_input_tensor = torch.nn.functional.pad(
            torch_input_tensor, (0, channel_padding_needed, 0, 0, 0, 0), value=0.0
        )
        c = min_channels
    torch_input_tensor = torch_input_tensor.reshape(1, 1, n * h * w, c)
    nhw = n * h * w
    shard_size = math.ceil((nhw / num_cores) / 32) * 32
    input_mem_config = ttnn.create_sharded_memory_config(
        [1, 1, shard_size * num_cores, c],
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
    )
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return tt_inputs_host, input_mem_config


def setup_cfg(config_file, weights_path, output_cfg_path=None, new_res=(256, 512)):
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)

    cfg.merge_from_file(config_file)

    new_height, new_width = new_res

    cfg.INPUT.MIN_SIZE_TEST = new_height
    cfg.INPUT.MAX_SIZE_TEST = new_width

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = (new_height, new_width)

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()

    return cfg


def annotate_video_offline(device, cfg, model, tid, input_height, input_width, video_path, output_path, length=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    frame_idx = 0
    t0 = time.time()
    avg_fps = 0

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            inputs, image_list = preprocessing(
                cfg,
                frame_bgr,
                aug=ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST),
                target_height=input_height,
                target_width=input_width,
            )

            host_tensor, input_mem_config = setup_l1_sharded_input(device, image_list.tensor)
            ttnn.copy_host_to_device_tensor(host_tensor, model.input_tensor, 0)

            t1 = time.time()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

            outputs = {
                "sem_seg_results": ttnn.to_torch(model.output0),
                "center_results": ttnn.to_torch(model.output1),
                "offset_results": ttnn.to_torch(model.output2),
            }

            _, sem_cls_resized = postprocessing(cfg, inputs, outputs, image_list, original_h=h, original_w=w)

            visualizer = Visualizer(frame_bgr[:, :, ::-1], metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
            sem_t = torch.as_tensor(sem_cls_resized, device="cpu")
            draw = visualizer.draw_sem_seg(sem_t)
            annotated_rgb = draw.get_image()
            annotated_bgr = annotated_rgb[:, :, ::-1]

            out.write(annotated_bgr)

            dt = time.time() - t1
            avg_fps += dt
            if frame_idx % 30 == 0:
                avg_fps /= 30
                print(f"[{frame_idx:6d}] Average FPS={avg_fps:.2f}Hz, Elapsed={time.time()-t0:.1f}s")
                avg_fps = 0
            frame_idx += 1
            if length is not None:
                if frame_idx > length:
                    break

    cap.release()
    out.release()
    print(f"Saved: {output_path}")


def annotate_video_offline_torch(video_path, config_file, weights_path, output_path, length=None):
    cfg = setup_cfg(config_file, weights_path)
    predictor = DefaultPredictor(cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    frame_idx = 0
    t0 = time.time()
    avg_fps = 0

    with torch.no_grad():
        while True:
            ok, im = cap.read()
            if not ok:
                break

            original_h, original_w = im.shape[:2]
            target_height, target_width = 256, 512
            im_resized = cv2.resize(im, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # Run inference
            t1 = time.time()
            outputs = predictor(im_resized)
            panoptic_seg_tensor, segments_info = outputs["panoptic_seg"]

            # Resize segmentation map back to original size
            panoptic_seg_np = panoptic_seg_tensor.to("cpu").numpy()
            panoptic_seg_resized = cv2.resize(
                panoptic_seg_np.astype(np.float32), (original_w, original_h), interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

            panoptic_seg_for_visualizer = torch.from_numpy(panoptic_seg_resized)

            # Visualization
            # metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            visualizer = Visualizer(im[:, :, ::-1], metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
            draw = visualizer.draw_panoptic_seg_predictions(panoptic_seg_for_visualizer, segments_info)
            annotated_bgr = draw.get_image()[:, :, ::-1]

            out.write(annotated_bgr)

            dt = time.time() - t1
            avg_fps += dt
            if frame_idx % 30 == 0:
                avg_fps = 30 / avg_fps
                print(f"[{frame_idx:6d}] Average FPS={avg_fps:.2f}Hz, Elapsed={time.time()-t0:.1f}s")
                avg_fps = 0
            frame_idx += 1
            if length is not None:
                if frame_idx > length:
                    break

    cap.release()
    out.release()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    args = parse_args()

    # ---- config / model init (same as your main) ----
    curr_file_path = os.path.dirname(os.path.realpath(__file__))
    CONFIG_FILE = os.path.join(
        curr_file_path,
        "reference/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml",
    )
    WEIGHTS_PATH = os.path.join(curr_file_path, "reference/pdl_weights.pt")

    in_vid = args.source
    out_vid = os.path.join(
        curr_file_path, "output", "torch_annotated_car.mp4" if args.torch else "ttnn_annotated_car.mp4"
    )
    os.makedirs(os.path.dirname(out_vid), exist_ok=True)

    if not args.torch:
        cfg = setup_cfg(CONFIG_FILE, WEIGHTS_PATH, new_res=(256, 512))

        l1_small_size = 10240 * 6
        device = ttnn.open_device(device_id=0, l1_small_size=l1_small_size, trace_region_size=10419200)
        device.enable_program_cache()
        # ttnn.device.EnablePersistentKernelCache()

        torch_model = PanopticDeepLab(cfg).eval()
        state = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
        torch_model.load_state_dict(state)
        params = create_pdl_model_parameters(torch_model)
        model = TtPanopticDeepLab(device, params, params["model_args"])

        H, W = args.image_height, args.image_width  # or argparse
        dummy = torch.rand(1, 3, H, W, dtype=torch.float32)
        model.input_tensor = preprate_ttnn_inputs(device, dummy)
        _ = model()
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        _ = model()
        ttnn.end_trace_capture(device, tid, cq_id=0)

        annotate_video_offline(
            device, cfg, model, tid, H, W, in_vid, out_vid, length=None if args.frames == 0 else args.frames
        )

        ttnn.close_device(device)

    else:
        annotate_video_offline_torch(
            in_vid, CONFIG_FILE, WEIGHTS_PATH, out_vid, length=None if args.frames == 0 else args.frames
        )
