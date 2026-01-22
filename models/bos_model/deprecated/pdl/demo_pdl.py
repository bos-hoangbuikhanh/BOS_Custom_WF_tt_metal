import argparse
import multiprocessing as mp
import os
import queue
import time

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.utils.visualizer import ColorMode, Visualizer

import ttnn
from models.bos_model.pdl.reference.norm_utils import ImageList
from models.bos_model.pdl.reference.panoptic_seg import PanopticDeepLab
from models.bos_model.pdl.reference.post_processing import (
    ResizeShortestEdge,
    get_panoptic_segmentation,
    sem_seg_postprocess,
)
from models.bos_model.pdl.tt.model_processing import create_pdl_model_parameters
from models.bos_model.pdl.tt.ttnn_panoptic_seg import TtPanopticDeepLab

# --------------------------------------------------------------------------------------
# Config & helpers
# --------------------------------------------------------------------------------------


def setup_cfg(config_file: str, weights_path: str, new_res=(256, 512)):
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST = new_res
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = new_res
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg


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


def postprocess(
    cfg,
    sem_seg_result,  # [C, H', W'] logits (torch.Tensor, CPU)
    center_result,  # [1, H', W'] (torch.Tensor, CPU)
    offset_result,  # [2, H', W'] (torch.Tensor, CPU)
    image_size_hw,  # (H_infer, W_infer) the model's processing size
    height,
    width,  # eval size (Detectron2 input_per_image height/width)
    original_h,
    original_w,
):
    """
    Returns:
        panoptic_resized_np: (H_orig, W_orig) int32 panoptic map (NumPy)
        sem_cls_resized_np:  (H_orig, W_orig) int32 semantic class map (NumPy)
    """
    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Resize raw outputs to eval size
    r = sem_seg_postprocess(sem_seg_result, image_size_hw, height, width)  # [C, H, W]
    c = sem_seg_postprocess(center_result, image_size_hw, height, width)  # [1, H, W]
    o = sem_seg_postprocess(offset_result, image_size_hw, height, width)  # [2, H, W]

    # Panoptic assembly at eval size
    panoptic_image, _ = get_panoptic_segmentation(
        r.argmax(dim=0, keepdim=True),  # [1, H, W] class ids
        c,
        o,
        thing_ids=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id.values(),
        label_divisor=meta.label_divisor,
        stuff_area=cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA,
        void_label=-1,
        threshold=cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD,
        nms_kernel=cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL,
        top_k=cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE,
    )
    panoptic_image = panoptic_image.squeeze(0)  # [H, W] torch.IntTensor (CPU)

    # --- Resize both maps back to the original size ---
    # Panoptic (nearest)
    pan_np = panoptic_image.cpu().numpy().astype(np.int32)
    panoptic_resized_np = cv2.resize(
        pan_np.astype(np.float32), (original_w, original_h), interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    # Semantic classes (nearest)
    sem_cls = r.argmax(dim=0).cpu().numpy().astype(np.int32)
    sem_cls_resized_np = cv2.resize(
        sem_cls.astype(np.float32), (original_w, original_h), interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    return panoptic_resized_np, sem_cls_resized_np


def colorize_semantic_overlay(sem_cls_resized, cfg):
    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    H, W = sem_cls_resized.shape
    black_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    vis = Visualizer(black_rgb, meta, scale=1.0, instance_mode=ColorMode.IMAGE)
    sem_t = torch.as_tensor(sem_cls_resized, device="cpu")
    overlay_bgr = vis.draw_sem_seg(sem_t).get_image()[:, :, ::-1].copy()
    return overlay_bgr


def postprocess_fast(
    meta,  # pass precomputed: MetadataCatalog.get(...)
    label_divisor,  # meta.label_divisor
    thing_ids,  # tuple/list(meta.thing_dataset_id_to_contiguous_id.values())
    cfg,  # for thresholds only (or pass those scalars instead)
    sem_seg_result,  # [C,H',W'] torch.float32 (CPU)
    center_result,  # [1,H',W'] torch.float32 (CPU)
    offset_result,  # [2,H',W'] torch.float32 (CPU)
    image_size_hw,  # (H_infer, W_infer)
    height,
    width,  # eval H,W
    original_h,
    original_w,
):
    # Resize to eval size (torch → torch). sem_seg_postprocess returns torch tensors already.
    r = sem_seg_postprocess(sem_seg_result, image_size_hw, height, width)  # [C,H,W]
    c = sem_seg_postprocess(center_result, image_size_hw, height, width)  # [1,H,W]
    o = sem_seg_postprocess(offset_result, image_size_hw, height, width)  # [2,H,W]

    # Compute semantic classes ONCE
    sem_cls_eval = r.argmax(dim=0, keepdim=True)  # [1,H,W], torch.int64

    # Panoptic assembly
    panoptic_image, _ = get_panoptic_segmentation(
        sem_cls_eval,
        c,
        o,
        thing_ids=thing_ids,
        label_divisor=label_divisor,
        stuff_area=cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA,
        void_label=-1,
        threshold=cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD,
        nms_kernel=cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL,
        top_k=cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE,
    )
    panoptic_image = panoptic_image.squeeze(0).to(torch.int32)  # [H,W], torch.int32 (CPU)

    # === Resize back to original with NEAREST, no float casts ===
    pan_np = panoptic_image.numpy()  # int32
    panoptic_resized_np = cv2.resize(pan_np, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    sem_cls_np = sem_cls_eval.squeeze(0).to(torch.int32).numpy()  # [H,W], int32
    sem_cls_resized_np = cv2.resize(sem_cls_np, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    return panoptic_resized_np, sem_cls_resized_np


def build_semantic_palette_from_meta(meta, num_classes=None):
    """
    Returns a (K,3) uint8 palette where row i is the color for class i.
    Falls back to a generated palette if meta lacks enough colors.
    """
    # Collect colors Detectron2-style
    colors = []
    if hasattr(meta, "thing_colors") and meta.thing_colors:
        colors.extend(meta.thing_colors)
    if hasattr(meta, "stuff_colors") and meta.stuff_colors:
        colors.extend(meta.stuff_colors)

    # Deduce number of classes if not provided
    K = num_classes or getattr(meta, "num_classes", None) or (max(len(colors), 1))

    # If fewer colors than K, repeat or generate
    if len(colors) < K:
        # simple deterministic palette extension
        rng = np.random.RandomState(0)
        extra = rng.randint(0, 255, size=(K - len(colors), 3), dtype=np.uint8)
        palette = np.vstack([np.array(colors, dtype=np.uint8), extra])
    else:
        palette = np.array(colors[:K], dtype=np.uint8)

    if palette.shape[0] < K:  # final guard
        pad = np.zeros((K - palette.shape[0], 3), dtype=np.uint8)
        palette = np.vstack([palette, pad])
    return palette  # (K,3) uint8


def colorize_semantic_overlay_fast(sem_cls_resized_np, palette):
    """
    sem_cls_resized_np: (H,W) int32
    palette: (K,3) uint8 color table (RGB or BGR as you prefer)
    Returns: (H,W,3) uint8 colored image (BGR if your palette is BGR)
    """
    # Clamp labels to palette range for safety
    max_label = palette.shape[0] - 1
    labels = np.clip(sem_cls_resized_np, 0, max_label)

    # Vectorized indexing: O(HW) in C
    overlay = palette[labels]  # (H,W,3) uint8
    return overlay


# --------------------------------------------------------------------------------------
# Inference worker: preprocess → device IO → execute TTNN trace → download logits
# --------------------------------------------------------------------------------------


def infer_worker(
    config_file, enable_persistent_cache, weights_path, H, W, result_queue, frame_queue, stop_event, device_mem
):
    """Lives in its own process: builds TTNN + model locally and streams results to result_queue."""
    torch.set_num_threads(1)

    cfg = setup_cfg(config_file, weights_path, new_res=(H, W))

    # TTNN device + model
    l1_small_size = device_mem.get("l1_small_size", 10240 * 6)
    trace_region_size = device_mem.get("trace_region_size", 10419200)

    device = ttnn.open_device(device_id=0, l1_small_size=l1_small_size, trace_region_size=trace_region_size)
    device.enable_program_cache()
    if enable_persistent_cache:
        ttnn.device.EnablePersistentKernelCache()

    try:
        # Optional persistent cache if available
        try:
            ttnn.device.EnablePersistentKernelCache()
        except Exception:
            pass

        torch_model = PanopticDeepLab(cfg).eval()
        state = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
        torch_model.load_state_dict(state)
        params = create_pdl_model_parameters(torch_model)
        model = TtPanopticDeepLab(device, params, params["model_args"])

        # Prepare input tensor shape once and capture trace
        dummy = torch.rand(1, 3, H, W, dtype=torch.float32)
        ttnn_input = dummy.permute((0, 2, 3, 1))  # NCHW → NHWC
        ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, device=device)
        model.input_tensor = ttnn.reshape(ttnn_input, (1, 1, 1 * H * W, 3))

        _ = model()  # warmup
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        _ = model()
        ttnn.end_trace_capture(device, tid, cq_id=0)

        aug = ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

        while not stop_event.is_set():
            try:
                frame_bgr = frame_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            t0 = time.time()
            inputs, image_list = preprocessing(cfg, frame_bgr, aug, target_height=H, target_width=W)

            # host→device copy into the traced input tensor
            torch_input_tensor = image_list.tensor.permute((0, 2, 3, 1))  # NCHW → NHWC
            n, h, w, c = torch_input_tensor.shape
            torch_input_tensor = torch_input_tensor.reshape(1, 1, n * h * w, c)
            tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(tt_inputs_host, model.input_tensor, 0)

            # execute trace (blocking)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

            # download outputs
            sem_t = ttnn.to_torch(model.output0)
            center_t = ttnn.to_torch(model.output1)
            offset_t = ttnn.to_torch(model.output2)

            # guard against bfloat16 → numpy limitations
            if sem_t.dtype == torch.bfloat16:
                sem_t = sem_t.float()
            if center_t.dtype == torch.bfloat16:
                center_t = center_t.float()
            if offset_t.dtype == torch.bfloat16:
                offset_t = offset_t.float()

            sem = sem_t.cpu().numpy().astype(np.float16)
            center = center_t.cpu().numpy().astype(np.float16)
            offset = offset_t.cpu().numpy().astype(np.float16)

            payload = {
                "sem": sem,
                "center": center,
                "offset": offset,
                "image_size": image_list.image_sizes[0],
                "height": inputs[0]["height"],
                "width": inputs[0]["width"],
                "orig_h": frame_bgr.shape[0],
                "orig_w": frame_bgr.shape[1],
            }

            # keep only freshest model result
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break
            result_queue.put(payload)

            dt = time.time() - t0
            # print(f"Model+pre/post IO FPS ≈ {1.0/dt:.2f} Hz (dt={dt:.3f}s)")
            print(f"Model Pre-process + Execution FPS ≈ {1.0/dt:.2f} Hz (Time={dt:.3f}s)")

    except Exception as e:
        try:
            result_queue.put({"error": str(e)})
        except Exception:
            pass
        raise
    finally:
        ttnn.close_device(device)
        # try:
        #     ttnn.close_device(device)
        # except Exception:
        #     pass


# --------------------------------------------------------------------------------------
# Visualization worker: postprocess logits → emit overlay only
# --------------------------------------------------------------------------------------


def viz_worker(config_file, execute_with_labels, result_queue, overlay_queue, stop_event):
    """
    Reads raw logits from result_queue, runs postprocessing, and emits a BGR overlay
    when (and only when) a new model output arrives. The main loop blends this overlay
    onto every frame so the mask persists between inferences.
    """
    torch.set_num_threads(1)
    cfg = setup_cfg(config_file, weights_path="/dev/null", new_res=(256, 512))

    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    palette_bgr = build_semantic_palette_from_meta(meta)[:, ::-1]  # if meta colors are RGB, flip to BGR once

    while not stop_event.is_set():
        try:
            payload = result_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        if "error" in payload:
            print("[viz] inference error:", payload["error"])
            continue

        post_process_start_time = time.time()

        sem = torch.from_numpy(payload["sem"]).float()
        center = torch.from_numpy(payload["center"]).float()
        offset = torch.from_numpy(payload["offset"]).float()
        image_size = payload["image_size"]
        height, width = payload["height"], payload["width"]
        orig_h, orig_w = payload["orig_h"], payload["orig_w"]

        if execute_with_labels:
            _, sem_cls_resized = postprocess(
                cfg,
                sem,
                center,
                offset,
                image_size,
                height,
                width,
                orig_h,
                orig_w,
            )
            overlay_bgr = colorize_semantic_overlay(sem_cls_resized, cfg)
        else:
            _, sem_cls_resized_np = postprocess_fast(
                meta=meta,
                label_divisor=meta.label_divisor,
                thing_ids=tuple(meta.thing_dataset_id_to_contiguous_id.values()),
                cfg=cfg,
                sem_seg_result=sem,
                center_result=center,
                offset_result=offset,
                image_size_hw=image_size,
                height=height,
                width=width,
                original_h=orig_h,
                original_w=orig_w,
            )
            overlay_bgr = colorize_semantic_overlay_fast(sem_cls_resized_np, palette_bgr)

        post_process_time = time.time() - post_process_start_time
        print(f"Post-processing Time = {post_process_time:.3f}")

        while not overlay_queue.empty():
            try:
                overlay_queue.get_nowait()
            except queue.Empty:
                break
        overlay_queue.put(overlay_bgr)


# --------------------------------------------------------------------------------------
# CLI / capture helpers
# --------------------------------------------------------------------------------------


def parse_args():
    curr_file_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            curr_file_path,
            "reference/configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml",
        ),
        help="Detectron2 config file path",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(curr_file_path, "reference/pdl_weights.pt"),
        help="Model weights path",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.path.join(curr_file_path, "videos/car.mp4"),
        help="Video source: webcam index (e.g., 0) or path/URL",
    )
    parser.add_argument("--labels", action="store_true", help="Execution wthout labels is much faster")
    parser.add_argument("--input_height", type=int, default=256)
    parser.add_argument("--input_width", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity [0..1]")
    parser.add_argument("--save", type=str, default="", help="Optional output video path")
    parser.add_argument("--fps", type=float, default=30.0, help="Target display FPS (for pacing & writer)")
    parser.add_argument("--device_l1_small", type=int, default=10240 * 6)
    parser.add_argument("--device_trace_region", type=int, default=10419200)
    parser.add_argument("--cap_width", type=int, default=1280)
    parser.add_argument("--cap_height", type=int, default=720)
    parser.add_argument("-p", "--persistent_cache", action="store_true", help="enable trace mode")
    return parser.parse_args()


def open_capture(src: str):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {src}")
    return cap


# --------------------------------------------------------------------------------------
# Main (capture + hold-last-overlay composition + display)
# --------------------------------------------------------------------------------------


def main():
    args = parse_args()

    mp.set_start_method("spawn", force=True)

    cap = open_capture(args.source)
    # Fix capture size for webcams that fluctuate
    if args.source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
        # Reduce camera buffering where supported
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    ok, test_frame = cap.read()
    if not ok:
        raise RuntimeError("No frames from source.")

    H0, W0 = test_frame.shape[:2]
    print(f"Input stream size: {W0}x{H0}")

    # Queues
    frame_queue = mp.Queue(maxsize=1)  # frames → inference worker
    result_queue = mp.Queue(maxsize=1)  # logits  → viz worker
    overlay_queue = mp.Queue(maxsize=1)  # overlays → main
    stop_event = mp.Event()

    # Launch workers
    p_infer = mp.Process(
        target=infer_worker,
        args=(
            args.config,
            args.persistent_cache,
            args.weights,
            int(args.input_height),
            int(args.input_width),
            result_queue,
            frame_queue,
            stop_event,
            {"l1_small_size": int(args.device_l1_small), "trace_region_size": int(args.device_trace_region)},
        ),
        daemon=True,
    )
    p_viz = mp.Process(
        target=viz_worker,
        args=(args.config, args.labels, result_queue, overlay_queue, stop_event),
        daemon=True,
    )

    p_infer.start()
    p_viz.start()

    # Optional writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (W0, H0))

    # Display pacing
    frame_interval = 1.0 / max(1e-6, args.fps)
    last_tick = time.time()

    # Overlay cache
    last_overlay = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Push latest frame to inference (non-blocking; drop stale)
            try:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put_nowait(frame_bgr)
            except (queue.Full, queue.Empty):
                pass

            # Pull newest overlay if available
            try:
                while True:
                    cand = overlay_queue.get_nowait()
                    last_overlay = cand
            except queue.Empty:
                pass

            # Compose
            show = frame_bgr
            if last_overlay is not None:
                if last_overlay.shape[:2] != frame_bgr.shape[:2]:
                    overlay_rs = cv2.resize(
                        last_overlay, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
                    )
                else:
                    overlay_rs = last_overlay
                mask = overlay_rs.sum(axis=2) > 0
                if mask.any():
                    out = show.copy()
                    out[mask] = cv2.addWeighted(
                        out[mask], 1.0 - float(args.alpha), overlay_rs[mask], float(args.alpha), 0.0
                    )
                    show = out

            if writer is not None:
                writer.write(show)

            cv2.imshow("Panoptic-DeepLab Live (holding last mask)", show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # pace display if needed
            now = time.time()
            dt = now - last_tick
            if dt < frame_interval:
                time.sleep(frame_interval - dt)
            last_tick = now

    finally:
        stop_event.set()
        p_infer.join(timeout=2.0)
        p_viz.join(timeout=2.0)
        # try:
        #     p_infer.join(timeout=2.0)
        # except Exception:
        #     pass
        # try:
        #     p_viz.join(timeout=2.0)
        # except Exception:
        #     pass
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
