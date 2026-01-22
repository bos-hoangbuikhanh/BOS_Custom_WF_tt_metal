# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from loguru import logger

# Conditional imports for TTNN functionality
try:
    from ttnn.model_preprocessing import preprocess_model_parameters

    import ttnn
    from models.bos_model.ufld_v2.tests.pcc.test_ttnn_ufld_v2 import create_custom_mesh_preprocessor, get_mesh_mappers
    from models.bos_model.ufld_v2.tt.ttnn_ufld_v2 import TtnnUFLDv2

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    print("Warning: TTNN not available. Only Torch mode will be functional.")

from models.bos_model.ufld_v2.demo import model_config as cfg
from models.bos_model.ufld_v2.demo.demo_utils import generate_tusimple_lines
from models.bos_model.ufld_v2.tt.model_preprocessing import UFLD_V2_L1_SMALL_SIZE, load_torch_model

# --------------------------------------------------------------------------------------
# Config & helpers
# --------------------------------------------------------------------------------------


def preprocessing(image, target_height=320, target_width=800, crop_ratio=1.0):
    """Preprocess image for UFLD v2 model."""
    img_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((int(target_height / crop_ratio), target_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    img = img_transforms(image)
    img = img[:, -target_height:, :]  # Crop
    return img


def postprocess(pred, row_anchor, col_anchor):
    """Process model output to generate lane predictions."""
    num_grid_row = cfg.num_cell_row
    num_cls_row = cfg.num_row
    num_lane_on_row = cfg.num_lanes
    num_grid_col = cfg.num_cell_col
    num_cls_col = cfg.num_col
    num_lane_on_col = cfg.num_lanes

    # Calculate dimensions
    dim1 = num_grid_row * num_cls_row * num_lane_on_row
    dim2 = num_grid_col * num_cls_col * num_lane_on_col
    dim3 = 2 * num_cls_row * num_lane_on_row
    dim4 = 2 * num_cls_col * num_lane_on_col

    pred_dict = {
        "loc_row": pred[:, :dim1].view(-1, num_grid_row, num_cls_row, num_lane_on_row),
        "loc_col": pred[:, dim1 : dim1 + dim2].view(-1, num_grid_col, num_cls_col, num_lane_on_col),
        "exist_row": pred[:, dim1 + dim2 : dim1 + dim2 + dim3].view(-1, 2, num_cls_row, num_lane_on_row),
        "exist_col": pred[:, -dim4:].view(-1, 2, num_cls_col, num_lane_on_col),
    }

    lanes = generate_tusimple_lines(
        pred_dict["loc_row"][0],
        pred_dict["exist_row"][0],
        pred_dict["loc_col"][0],
        pred_dict["exist_col"][0],
        row_anchor=row_anchor,
        col_anchor=col_anchor,
        mode="4row",
    )

    return lanes


def draw_lanes(image, lanes, h_samples=None, mask_constant=-2):
    """Draw lane predictions on image."""
    # Define h_samples if not provided (keeping the original list for completeness)
    if h_samples is None:
        h_samples = [
            160,
            170,
            180,
            190,
            200,
            210,
            220,
            230,
            240,
            250,
            260,
            270,
            280,
            290,
            300,
            310,
            320,
            330,
            340,
            350,
            360,
            370,
            380,
            390,
            400,
            410,
            420,
            430,
            440,
            450,
            460,
            470,
            480,
            490,
            500,
            510,
            520,
            530,
            540,
            550,
            560,
            570,
            580,
            590,
            600,
            610,
            620,
            630,
            640,
            650,
            660,
            670,
            680,
            690,
            700,
            710,
        ]

    img = image.copy()
    h_samples = np.array(h_samples)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]  # green, red, blue, yellow
    thickness = 5
    radius = 5

    for lane_idx, lane_xs in enumerate(lanes):
        lane_xs = np.array(lane_xs)
        valid_mask = lane_xs != mask_constant
        valid_xs = lane_xs[valid_mask]
        valid_ys = h_samples[valid_mask]

        points = np.array([valid_xs, valid_ys]).T.astype(np.int32)

        if len(points) >= 2:
            cv2.polylines(img, [points], isClosed=False, color=colors[lane_idx % len(colors)], thickness=thickness)

            for x, y in points:
                cv2.circle(img, (x, y), radius=radius, color=colors[lane_idx % len(colors)], thickness=-1)

    return img


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args():
    curr_file_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(curr_file_path, "reference/tusimple_34.pth"),
        help="Model weights path",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(curr_file_path, "images/1492626760788443246_0_2.jpg"),
        help="Input image path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/bos_model/ufld_v2/demo/outputs",
        help="Output image path (if not specified, will display only)",
    )
    parser.add_argument("--input_height", type=int, default=320)
    parser.add_argument("--input_width", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity [0..1]")
    parser.add_argument(
        "--backend",
        type=str,
        default="ttnn",
        choices=["ttnn", "torch"],
        help="Model backend to use: 'ttnn' for Tenstorrent or 'torch' for PyTorch (CPU/GPU)",
    )
    # TTNN specific arguments
    if TTNN_AVAILABLE:
        parser.add_argument("--device_l1_small", type=int, default=UFLD_V2_L1_SMALL_SIZE)
        parser.add_argument("--device_trace_region", type=int, default=10419200)
        parser.add_argument("-p", "--persistent_cache", action="store_true", help="enable persistent cache")
    return parser.parse_args()


def run_ufldv2_demo(args, device):
    if args.backend == "ttnn" and not TTNN_AVAILABLE:
        raise RuntimeError(
            "TTNN backend requested, but ttnn package is not available. Please install it or use '--backend torch'."
        )

    # Check if a TTNN device is provided for TTNN mode
    if args.backend == "ttnn" and device is None:
        raise ValueError("TTNN device object must be provided when running with '--backend ttnn'")

    if args.backend == "ttnn" and args.persistent_cache:
        try:
            ttnn.device.EnablePersistentKernelCache()
        except Exception as e:
            print(f"Warning: Could not enable persistent cache: {e}")

    try:
        # Load torch model (used for both backends, but TTNN converts it)
        print("Loading model...")
        torch_model = load_torch_model(None, use_pretrained_weight=True)
        torch_model.eval()

        # --- Model Setup based on Backend ---
        model = None
        ttnn_device = device

        if args.backend == "ttnn":
            # --- TTNN Setup ---
            # Create dummy input for inference args
            H, W = args.input_height, args.input_width
            batch_size = args.batch_size
            dummy_input = torch.randn((batch_size, 3, H, W))

            # Preprocess model parameters
            print("Preprocessing TTNN parameters...")
            _, weights_mesh_mapper, _ = get_mesh_mappers(ttnn_device)
            parameters = preprocess_model_parameters(
                initialize_model=lambda: torch_model,
                custom_preprocessor=create_custom_mesh_preprocessor(mesh_mapper=weights_mesh_mapper),
                device=ttnn_device,
            )

            # Infer conv args
            from ttnn.model_preprocessing import infer_ttnn_module_args

            parameters.conv_args = {}
            parameters.conv_args = infer_ttnn_module_args(
                model=torch_model, run_model=lambda model: torch_model(dummy_input), device=ttnn_device
            )

            # Create TTNN model
            print("Creating TTNN model...")
            model = TtnnUFLDv2(conv_args=parameters.conv_args, conv_pth=parameters, device=ttnn_device)

        elif args.backend == "torch":
            # --- Torch Setup ---
            model = torch_model
            # Optionally move to CUDA if available for faster torch execution
            if torch.cuda.is_available():
                print("Moving Torch model to CUDA device.")
                model.cuda()
            else:
                print("Using Torch model on CPU.")

        # Import get_test_loader from demo_utils
        from tqdm import tqdm

        from models.bos_model.ufld_v2.demo.demo_utils import get_test_loader

        image_dir = os.path.dirname(args.image)
        data_root = os.path.dirname(image_dir)

        # Create data loader (following demo_utils.py logic)
        print("Creating data loader...")
        H, W = args.input_height, args.input_width
        batch_size = args.batch_size
        loader = get_test_loader(
            batch_size=batch_size,
            data_root=data_root,  # Pass parent directory, loader will append "images"
            dataset="Tusimple",
            distributed=False,
            crop_ratio=cfg.crop_ratio,
            train_width=W,
            train_height=H,
        )

        # Get first batch for trace setup (TTNN only)
        if args.backend == "ttnn":
            print("Loading first batch for trace setup...")
            data_iter = iter(loader)
            imgs, names = next(data_iter)

            print(f"Processing image: {names[0]}")
            print(f"Input tensor shape: {imgs.shape}")  # Should be [1, 3, 320, 800]

            # NHWC format
            ttnn_input = ttnn.from_torch(imgs, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=ttnn_device)

            # Set the input tensor on the model
            model.input_tensor = ttnn_input

            # Warmup with real data
            print("Warming up...")
            _ = model()

            # Capture trace
            print("Capturing trace...")
            tid = ttnn.begin_trace_capture(ttnn_device, cq_id=0)
            output = model()
            ttnn.end_trace_capture(ttnn_device, tid, cq_id=0)
            print("Trace captured successfully!")

            # Reset the data loader for full run
            loader = get_test_loader(
                batch_size=batch_size,
                data_root=data_root,
                dataset="Tusimple",
                distributed=False,
                crop_ratio=cfg.crop_ratio,
                train_width=W,
                train_height=H,
            )

        # Configure anchors
        cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
        h_samples = [
            160,
            170,
            180,
            190,
            200,
            210,
            220,
            230,
            240,
            250,
            260,
            270,
            280,
            290,
            300,
            310,
            320,
            330,
            340,
            350,
            360,
            370,
            380,
            390,
            400,
            410,
            420,
            430,
            440,
            450,
            460,
            470,
            480,
            490,
            500,
            510,
            520,
            530,
            540,
            550,
            560,
            570,
            580,
            590,
            600,
            610,
            620,
            630,
            640,
            650,
            660,
            670,
            680,
            690,
            700,
            710,
        ]

        # Calculate dimensions for postprocessing
        num_grid_row = cfg.num_cell_row
        num_cls_row = cfg.num_row
        num_lane_on_row = cfg.num_lanes
        num_grid_col = cfg.num_cell_col
        num_cls_col = cfg.num_col
        num_lane_on_col = cfg.num_lanes

        dim1 = num_grid_row * num_cls_row * num_lane_on_row
        dim2 = num_grid_col * num_cls_col * num_lane_on_col
        dim3 = 2 * num_cls_row * num_lane_on_row
        dim4 = 2 * num_cls_col * num_lane_on_col

        # Process all images from the loader
        print("Processing images...")
        image_count = 0
        total_inference_time = 0
        total_time_absolute = 0
        for data in tqdm(loader, desc="Processing images"):
            start_time = time.time()
            imgs, names = data

            # --- Inference based on Backend ---
            out_t = None
            if args.backend == "ttnn":
                # Update input tensor with new batch
                tt_inputs_host = ttnn.from_torch(imgs, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                ttnn.copy_host_to_device_tensor(tt_inputs_host, model.input_tensor, 0)

                # Execute trace
                t0 = time.time()
                ttnn.execute_trace(ttnn_device, tid, cq_id=0, blocking=True)
                inference_time = time.time() - t0
                total_inference_time += inference_time
                image_count += 1

                # Download output
                out_t = ttnn.to_torch(output).squeeze(dim=0).squeeze(dim=0)

                # Guard against bfloat16 → numpy limitations
                if out_t.dtype == torch.bfloat16:
                    out_t = out_t.float()

            elif args.backend == "torch":
                # Move input to GPU if model is on GPU
                input_tensor = imgs
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()

                # Run torch model
                t0 = time.time()
                with torch.no_grad():
                    raw_output = model(input_tensor)
                inference_time = time.time() - t0
                total_inference_time += inference_time
                image_count += 1

                # --- FIX APPLIED HERE ---
                # Check if the output is a tuple (e.g., due to auxiliary losses)
                if isinstance(raw_output, tuple):
                    # We assume the first element of the tuple is the final prediction tensor.
                    output_tensor_for_postprocess = raw_output[0]
                else:
                    output_tensor_for_postprocess = raw_output

                # Move output to CPU for post-processing
                # out_t = raw_output.cpu().squeeze(dim=0) # <-- ORIGINAL FAULTY LINE
                out_t = output_tensor_for_postprocess.cpu().squeeze(dim=0)
                # --- END FIX ---

            if out_t is None:
                continue

            # Reshape to match expected format [1, 39576]
            out = out_t.reshape(batch_size, -1)

            # Create prediction dictionary (matching demo_utils.py format)
            pred = {
                "loc_row": out[:, :dim1].view(-1, num_grid_row, num_cls_row, num_lane_on_row),
                "loc_col": out[:, dim1 : dim1 + dim2].view(-1, num_grid_col, num_cls_col, num_lane_on_col),
                "exist_row": out[:, dim1 + dim2 : dim1 + dim2 + dim3].view(-1, 2, num_cls_row, num_lane_on_row),
                "exist_col": out[:, -dim4:].view(-1, 2, num_cls_col, num_lane_on_col),
            }

            # Process each image in batch
            for b_idx, name in enumerate(names):
                # Generate lanes
                lanes = generate_tusimple_lines(
                    pred["loc_row"][b_idx],
                    pred["exist_row"][b_idx],
                    pred["loc_col"][b_idx],
                    pred["exist_col"][b_idx],
                    row_anchor=cfg.row_anchor,
                    col_anchor=cfg.col_anchor,
                    mode="4row",
                )

                total_time_absolute += time.time() - start_time

                print(f"\nImage: {name}")
                print(f"Detected {len(lanes)} lanes")

                # Load original image for visualization
                image_path = os.path.join(data_root, "images", name)
                image_bgr = cv2.imread(image_path)

                if image_bgr is not None:
                    # Draw lanes on original image
                    result = draw_lanes(image_bgr, lanes, h_samples)

                    # Blend with original
                    final = cv2.addWeighted(image_bgr, 1.0 - args.alpha, result, args.alpha, 0)

                    # Save output
                    if args.output:
                        output_dir = os.path.join(
                            args.output, f"{args.backend}_outputs"
                        )  # Save to a backend-specific folder
                        os.makedirs(output_dir, exist_ok=True)

                        output_name = f"{os.path.splitext(name)[0]}.jpg"
                        output_path = os.path.join(output_dir, output_name)

                        os.makedirs(output_dir, exist_ok=True)
                        cv2.imwrite(output_path, final)
                        print(f"Result saved to: {output_path}")

        # Print statistics
        total_fps = image_count / total_time_absolute if total_time_absolute > 0 else 0
        avg_inference_time = total_inference_time / image_count if image_count > 0 else 0

        print(f"\n{'='*60}")
        print(f"Backend: **{args.backend.upper()}**")
        print(f"Total images processed: {image_count}")
        print(f"Average inference time: {avg_inference_time:.3f}s")
        print(f"Average FPS (Inference Only): {1.0/avg_inference_time:.2f}" if avg_inference_time > 0 else "N/A")
        print(f"\n**Total End-to-End FPS: {total_fps:.2f}**")
        print(f"{'='*60}")

    finally:
        # Close TTNN device only if it was opened
        if args.backend == "ttnn" and ttnn_device is not None:
            ttnn.close_device(ttnn_device)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def main():
    args = parse_args()
    device = None

    if args.backend == "ttnn":
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN backend selected but the package is not installed.")

        # Setup device for TTNN
        device = ttnn.open_device(
            device_id=0, l1_small_size=args.device_l1_small, trace_region_size=args.device_trace_region
        )
        if args.persistent_cache:
            try:
                ttnn.device.EnablePersistentKernelCache()
            except Exception as e:
                print(f"Warning: Could not enable persistent cache: {e}")

    # Run the demo
    print(f"=== Running the UFLDv2 demo with {args.backend.upper()} backend ===")
    run_ufldv2_demo(args, device)

    # Close TTNN device if opened outside of run_ufldv2_demo (for safety, though it's closed inside too)
    if args.backend == "ttnn" and device is not None:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
