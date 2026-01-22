import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.bos_model.ufld_v2.common import UFLD_V2_L1_SMALL_SIZE
from models.bos_model.ufld_v2.demo import model_config as cfg
from models.bos_model.ufld_v2.demo import model_config_culane_res34 as cfg_culane_res34
from models.bos_model.ufld_v2.demo.demo_utils import (
    generate_culane_lines,
    generate_tusimple_lines,
    overlay_lanes_on_frame,
    preprocess_frame,
)
from models.bos_model.ufld_v2.runner.performant_runner import UFLDPerformantRunner


def culane_ready(dataset_root: str) -> bool:
    required = [
        "driver_100_30frame",
        "driver_193_90frame",
        "driver_37_30frame",
        "list",
    ]
    if not os.path.isdir(dataset_root):
        return False
    return all(os.path.isdir(os.path.join(dataset_root, d)) for d in required)


def tusimple_ready(dataset_root: str, json_file: str) -> bool:
    if not os.path.isdir(dataset_root):
        return False
    return os.path.isfile(json_file)


def load_tusimple_image_list(dataset_root: str, json_file: str) -> list[str]:
    image_paths = []
    logger.info(f"[tusimple_dataset] dataset_root={dataset_root}, json_file={json_file}")

    with open(json_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rel = data.get("raw_file", None)
            if not rel:
                continue
            rel = rel.lstrip("/")
            abs_path = os.path.join(dataset_root, rel)
            if not os.path.exists(abs_path):
                logger.warning(f"[tusimple_dataset] image not found: {abs_path}")
                continue
            image_paths.append(abs_path)

    logger.info(f"[tusimple_dataset] total valid images: {len(image_paths)}")
    return image_paths


def load_culane_image_list(dataset_root: str, list_file: str) -> list[str]:
    image_paths = []

    logger.info(f"[culane_dataset] dataset_root={dataset_root}, list_file={list_file}")

    with open(list_file, "r") as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue

            # Remove a leading "/" if present
            rel = rel.lstrip("/")

            # First candidate: typical CULane directory structure
            cand_a = os.path.join(dataset_root, rel)

            # Second candidate: structure with an extra nested driver_xxx_30frame directory
            parts = rel.split("/")
            cand_b = None
            if len(parts) > 1:
                cand_b = os.path.join(dataset_root, parts[0], rel)

            # Choose the first path that actually exists
            if os.path.exists(cand_a):
                abs_path = cand_a
            elif cand_b is not None and os.path.exists(cand_b):
                abs_path = cand_b
            else:
                logger.warning(
                    f"[culane_dataset] path not found for rel='{rel}' " f"(cand_a='{cand_a}', cand_b='{cand_b}')"
                )
                continue

            image_paths.append(abs_path)

    logger.info(f"[culane_dataset] total valid images: {len(image_paths)}")
    return image_paths


def draw_text_with_box(
    img,
    text: str,
    org: tuple[int, int],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.9,
    text_color=(0, 255, 0),
    thickness: int = 2,
    bg_color=(0, 0, 0),
    padding: int = 4,
):
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = org
    x0 = x - padding
    y0 = y - text_h - padding
    x1 = x + text_w + padding
    y1 = y + baseline + padding

    # background box
    cv2.rectangle(
        img,
        (x0, y0),
        (x1, y1),
        bg_color,
        thickness=-1,
    )

    # text
    cv2.putText(
        img,
        text,
        org,
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )

    return img


def get_dataset_bundle(dataset: str):
    if dataset == "culane":
        base_w, base_h = 1640.0, 590.0
        h_samples = [250 + 10 * i for i in range(35)]
        cfg_culane_res34.row_anchor = np.linspace(0.42, 1.0, cfg_culane_res34.num_row).tolist()
        cfg_culane_res34.col_anchor = np.linspace(0, 1, cfg_culane_res34.num_col)
        model_type = "culane"
        H, W = cfg_culane_res34.train_height, cfg_culane_res34.train_width  # (320, 1600)
        return cfg_culane_res34, model_type, (H, W), base_w, base_h, h_samples

    if dataset == "tusimple":
        base_w, base_h = 1280.0, 720.0
        h_samples = [160 + 10 * i for i in range(56)]  # 160..710
        cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / base_h
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
        model_type = "tusimple"
        H, W = cfg.train_height, cfg.train_width  # (320, 800)
        return cfg, model_type, (H, W), base_w, base_h, h_samples

    raise ValueError(dataset)


def run_video_demo_with_runner(
    device,
    dataset="culane",  # (tusimple|culane)
    mode="video",  # "video" | "image"
    image_path=None,
    video_path=None,
    input_mode="dram_interleaved",
    max_frames=None,
    draw_mode="dot",
    loop=True,
    window_name="UFLD V2 CULane Demo (TT)",
    model_location_generator=None,
    dataset_root=None,
    list_file=None,  # for culane
    json_file=None,  # for tusimple
    output_path=None,
):
    """
    Refactored:
      - image/video share the same processing path
      - runner.run() is called only inside infer()
    """

    assert device is not None, "TT device is required"

    cfg, model_type, (H, W), base_w, base_h, h_samples = get_dataset_bundle(dataset)

    # ===== runner init =====
    dummy_input = torch.randn(1, 3, H, W)
    runner = None
    try:
        logger.info(f"Creating UFLDPerformantRunner (dataset={dataset}, input_mode={input_mode}, resolution=({H},{W}))")
        runner = UFLDPerformantRunner(
            device=device,
            model_location_generator=model_location_generator,
            device_batch_size=1,
            resolution=(H, W),
            torch_input_tensor=dummy_input,
            input_mode=input_mode,
            model_type=dataset,
        )

        # ===== infer: runner output(flat) -> pred dict (single place) =====
        def infer(imgs):
            out_dev = runner.run(imgs)
            out = ttnn.to_torch(out_dev, mesh_composer=runner.runner_infra.output_mesh_composer).squeeze(1).squeeze(1)
            out = out.reshape(1, -1)

            num_grid_row = cfg.num_cell_row
            num_cls_row = cfg.num_row
            num_lane = cfg.num_lanes
            num_grid_col = cfg.num_cell_col
            num_cls_col = cfg.num_col

            dim1 = num_grid_row * num_cls_row * num_lane
            dim2 = num_grid_col * num_cls_col * num_lane
            dim3 = 2 * num_cls_row * num_lane
            dim4 = 2 * num_cls_col * num_lane

            pred = {
                "loc_row": out[:, :dim1].view(-1, num_grid_row, num_cls_row, num_lane),
                "loc_col": out[:, dim1 : dim1 + dim2].view(-1, num_grid_col, num_cls_col, num_lane),
                "exist_row": out[:, dim1 + dim2 : dim1 + dim2 + dim3].view(-1, 2, num_cls_row, num_lane),
                "exist_col": out[:, -dim4:].view(-1, 2, num_cls_col, num_lane),
            }
            return pred

        # ===== render: pred -> overlay (CULane fixed) =====
        def render(frame, pred, dataset, cfg, h_samples, base_w, base_h, draw_mode="dot"):
            if dataset == "tusimple":
                lanes = generate_tusimple_lines(
                    pred["loc_row"][0].detach().cpu(),
                    pred["exist_row"][0].detach().cpu(),
                    pred["loc_col"][0].detach().cpu(),
                    pred["exist_col"][0].detach().cpu(),
                    row_anchor=cfg.row_anchor,
                    col_anchor=cfg.col_anchor,
                    mode="4row",  # other modes: "2row2col",, "4col"
                )
            elif dataset == "culane":
                lanes = generate_culane_lines(
                    row_out=pred["loc_row"][0].detach().cpu(),
                    row_ext=pred["exist_row"][0].detach().cpu(),
                    col_out=pred["loc_col"][0].detach().cpu(),
                    col_ext=pred["exist_col"][0].detach().cpu(),
                    row_anchor=cfg.row_anchor,
                    col_anchor=cfg.col_anchor,
                    h_samples=h_samples,
                    img_w=base_w,
                    img_h=base_h,
                    mode="2row2col",  # other modes: "4row", "4col"
                )
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            overlay = overlay_lanes_on_frame(
                frame,
                {"lanes": lanes, "h_samples": h_samples},
                radius=4,
                point_length=1,
                mask_constant=-2,
                y_min_ratio=0.3,
                y_max_ratio=1.05,
                dataset=dataset,
                draw_mode=draw_mode,
            )
            return overlay

        def process_frame(frame):
            # ---- Tusimple only: crop ROI to 1280x720 bottom-center ----
            if dataset == "tusimple":
                H_full, W_full = frame.shape[:2]
                target_w, target_h = 1280, 720

                # Safely clamp values
                target_w = min(target_w, W_full)
                target_h = min(target_h, H_full)

                x0 = max(0, (W_full - target_w) // 2)
                y0 = max(0, H_full - target_h)
                x1 = min(W_full, x0 + target_w)
                y1 = min(H_full, y0 + target_h)

                roi = frame[y0:y1, x0:x1, :].copy()
                frame_for_model = roi
            else:
                # ---- others: use full frame ----
                x0 = y0 = 0
                y1, x1 = frame.shape[:2]
                roi = frame
                frame_for_model = frame

            # ---- preprocess -> infer ----
            imgs = preprocess_frame(frame_for_model, train_width=W, train_height=H, crop_ratio=cfg.crop_ratio)

            t0 = time.time()
            pred = infer(imgs)
            t1 = time.time()
            infer_fps = 1.0 / max(t1 - t0, 1e-6)

            # ---- render on the same frame we fed (roi or full) ----
            overlay_in = render(
                frame_for_model,
                pred,
                dataset=dataset,
                cfg=cfg,
                h_samples=h_samples,
                base_w=base_w,
                base_h=base_h,
                draw_mode=draw_mode,
            )

            # ---- if tusimple, paste ROI overlay back to full frame + draw ROI box ----
            if dataset == "tusimple":
                overlay = frame.copy()
                overlay[y0:y1, x0:x1, :] = overlay_in

                # ROI box
                cv2.rectangle(
                    overlay,
                    (x0, y0),
                    (x1 - 1, y1 - 1),
                    (0, 255, 255),
                    2,
                )
            else:
                overlay = overlay_in

            overlay = draw_text_with_box(
                overlay,
                f"infer_fps: {infer_fps:.2f}",
                org=(10, 30),
            )

            return overlay

        # ===== iterators =====
        def iter_single_image(path):
            frame = cv2.imread(path)
            if frame is None:
                raise RuntimeError(f"Failed to load image: {path}")
            yield frame

        def iter_video_frames(path, loop=True):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {path}")
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        if loop:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        break
                    yield frame
            finally:
                cap.release()

        def iter_image_list(image_paths, loop=True):
            if len(image_paths) == 0:
                raise RuntimeError("[dataset] no valid images found")

            while True:
                for p in image_paths:
                    frame = cv2.imread(p)
                    if frame is None:
                        logger.warning(f"[dataset] failed to read: {p}")
                        continue
                    yield frame
                if not loop:
                    break

        def run_display_loop(frame_iter, max_frames=None):
            prev_time = time.time()
            shown = 0

            for frame in frame_iter:
                overlay = process_frame(frame)

                now = time.time()
                disp_fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                overlay = draw_text_with_box(
                    overlay,
                    f"display_fps: {disp_fps:.2f}",
                    org=(10, 65),
                )

                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    cv2.waitKey(0)

                shown += 1
                if max_frames is not None and shown >= max_frames:
                    break

            cv2.destroyAllWindows()

        def run_video_save(
            video_path: str,
            output_path: str,
            process_frame,
            max_frames: int | None = None,
        ):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")

            # Input video properties
            in_fps = cap.get(cv2.CAP_PROP_FPS)
            if not in_fps or in_fps <= 1e-3:
                in_fps = 30.0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, in_fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

            # Determine total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = None

            # Enforce deterministic save length
            if total_frames is None and max_frames is None:
                cap.release()
                writer.release()
                raise ValueError("Total frame count is unknown. Please specify max_frames for video saving.")

            if max_frames is not None and total_frames is not None:
                target_frames = min(max_frames, total_frames)
            else:
                target_frames = max_frames or total_frames

            logger.info(f"[video_save] target_frames={target_frames}, " f"total_frames={total_frames}")

            try:
                written = 0
                for _ in tqdm(range(target_frames), desc="Saving video", unit="frame"):
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"[video_save] Early EOF at frame {written}/{target_frames}")
                        break

                    overlay = process_frame(frame)

                    if overlay.shape[1] != width or overlay.shape[0] != height:
                        overlay = cv2.resize(overlay, (width, height))

                    writer.write(overlay)
                    written += 1

                logger.info(f"[video_save] wrote {written} frames -> {output_path}")

            finally:
                writer.release()
                cap.release()

        # ===== mode dispatch =====
        if mode == "image":
            if image_path is None:
                raise ValueError("mode=image requires image_path")
            run_display_loop(iter_single_image(image_path), max_frames=1)
        elif mode == "video":
            if video_path is None:
                raise ValueError("mode=video requires video_path")
            run_display_loop(iter_video_frames(video_path, loop=loop), max_frames=max_frames)
        elif mode == "video_save":
            if video_path is None:
                raise ValueError("mode=video_save requires video_path")
            if output_path is None:
                raise ValueError("mode=video_save requires output_path")

            run_video_save(
                video_path=video_path,
                output_path=output_path,
                process_frame=process_frame,
                max_frames=max_frames,
            )

        elif mode == "dataset":
            if dataset_root is None:
                raise ValueError("mode=gallery requires dataset_root")

            if dataset == "culane":
                if list_file is None:
                    raise ValueError("dataset=culane requires --list_file")
                paths = load_culane_image_list(dataset_root, list_file)

            elif dataset == "tusimple":
                if json_file is None:
                    raise ValueError("dataset=tusimple requires --json_file")
                paths = load_tusimple_image_list(dataset_root, json_file)

            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

            run_display_loop(iter_image_list(paths, loop=loop), max_frames=max_frames)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    finally:
        if runner is not None:
            runner.release()


def main():
    import argparse

    import ttnn

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["image", "video", "dataset", "video_save"], default="video")

    p.add_argument("--image_path", default=None)
    p.add_argument("--video_path", default=None)
    p.add_argument("--output_path", default="output.mp4")

    p.add_argument("--dataset", choices=["culane", "tusimple"], required=True)
    p.add_argument("--list_file", default=None, help="CULane list/test.txt path")
    p.add_argument("--json_file", default=None, help="TuSimple test_label.json path")

    p.add_argument(
        "--input_mode",
        default="dram_interleaved",
        choices=[
            # "dram_sharded",
            "dram_interleaved",
            # "dram_interleaved_double_buffer"
        ],
    )
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--draw_mode", choices=["dot", "line"], default="dot")
    p.add_argument("--loop", action="store_true", help="Loop input when it ends")
    args = p.parse_args()

    # ------------------------------------------------------------
    # Prepare dataset symlinks if required files are missing.
    # This is only relevant for --mode dataset.
    # ------------------------------------------------------------
    dataset_root = None
    if args.mode == "dataset":
        image_data_root = os.path.join(cfg.data_root, "image_data")
        dataset_root = os.path.join(image_data_root, args.dataset)

        # Fill defaults if user did not pass explicit paths
        if args.dataset == "culane":
            if args.list_file is None:
                args.list_file = os.path.join(dataset_root, "list", "test.txt")

            if (not culane_ready(dataset_root)) or (not os.path.exists(args.list_file)):
                subprocess.run(
                    [sys.executable, os.path.join(cfg.data_root, "data_download.py"), "--culane"],
                    check=True,
                )
            # re-check
            if not culane_ready(dataset_root):
                raise RuntimeError(f"CULane dataset is still not ready under: {dataset_root}")

        elif args.dataset == "tusimple":
            if args.json_file is None:
                # Keep the existing CLI contract: json_file points to the tasks/label json.
                # If your downloader creates a compatibility symlink named "test_label.json",
                # this path is correct.
                args.json_file = os.path.join(dataset_root, "test_label.json")

            if not tusimple_ready(dataset_root, args.json_file):
                subprocess.run(
                    [sys.executable, os.path.join(cfg.data_root, "data_download.py"), "--tusimple"],
                    check=True,
                )

            if not tusimple_ready(dataset_root, args.json_file):
                raise RuntimeError(f"TuSimple dataset is still not ready under: {dataset_root}")

        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

    # --- Create TT device (same pattern as your other runner demos) ---
    device_dict = {
        "device_id": 0,
        "l1_small_size": UFLD_V2_L1_SMALL_SIZE,
        "trace_region_size": 23887872,
        "num_command_queues": 2,
    }
    ttnn_device = ttnn._ttnn.device
    device = ttnn_device.CreateDevice(**device_dict)

    try:
        run_video_demo_with_runner(
            device=device,
            dataset=args.dataset,
            mode=args.mode,
            image_path=args.image_path,
            video_path=args.video_path,
            input_mode=args.input_mode,
            max_frames=args.max_frames,
            draw_mode=args.draw_mode,
            loop=args.loop,
            dataset_root=dataset_root,
            list_file=args.list_file,
            json_file=args.json_file,
            output_path=args.output_path,
        )
    finally:
        # If you have an explicit device close/destroy utility in your repo, call it here.
        pass


if __name__ == "__main__":
    main()
