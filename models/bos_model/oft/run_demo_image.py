# for display
import argparse
import multiprocessing

# import matplotlib
# matplotlib.use("QtAgg")
import cv2
import time
from datetime import datetime

import os
import torch
import ttnn
import matplotlib.pyplot as plt
from loguru import logger

from models.bos_model.oft.reference.bbox import visualize_objects_cv
from models.bos_model.oft.reference.encoder import ObjectEncoder
from models.bos_model.oft.reference.oftnet import OftNet
from models.bos_model.oft.reference.utils import (
    get_abs_and_relative_error,
    load_calib,
    load_image,
    make_grid,
    visualize_score,
)
from models.bos_model.oft.reference.utils import print_object_comparison
from models.bos_model.oft.tests.common import (
    GRID_HEIGHT,
    GRID_RES,
    GRID_SIZE,
    H_PADDED,
    NMS_THRESH,
    W_PADDED,
    Y_OFFSET,
    load_checkpoint,
    visualize_tensor_distributions,
)
from models.bos_model.oft.tt.model_preprocessing import create_OFT_model_parameters, create_decoder_model_parameters
from models.bos_model.oft.tt.model_configs import ModelOptimizations
from models.bos_model.oft.tt.tt_oftnet import TTOftNet
from models.bos_model.oft.tt.tt_encoder import TTObjectEncoder
from models.bos_model.oft.tt.tt_resnet import TTBasicBlock
from tests.ttnn.utils_for_testing import check_with_pcc
from tests.ttnn.unit_tests.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", "--num_iter", type=int, default=10, help="number of iteration to process. -1 is infinite loop"
    )
    parser.add_argument("--torch", default=False, action="store_true", help="run torch model")
    # parser.add_argument("--trace", default=False, action="store_true", help="enable trace mode")  # not implemented
    # parser.add_argument("--save_result", default=False, action="store_true", help="save result images")

    args, _ = parser.parse_known_args(argv)
    return args


def postprocess_and_save_results(
    model_type,
    image,
    calib,
    grid,
    pred_encoded,
    objects,
    encoder,
    output_base="",
    res=(180, 540),
):
    # Decode predictions
    pred_encoded = [t[0].cpu() for t in pred_encoded]
    detections = encoder.decode(*pred_encoded, grid.cpu())

    img_out = visualize_objects_cv(image, calib, detections)

    if output_base:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = os.path.join(output_base, model_type)
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, f"result_{timestamp}_res{res[0]}x{res[1]}.jpg")
        cv2.imwrite(save_path, img_out)
        print(f"Predictions saved to: {save_path}")
    return img_out


def cv2_worker(lock, shared_dict):
    while True:
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


def main(device, model_location_generator, args):
    skip_if_not_blackhole_20_cores(device)
    device.disable_and_clear_program_cache()  # test hangs without this line on P150

    # Create output directory for saving visualizations
    output_dir = os.path.join(os.path.dirname(__file__), "demo/outputs")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    input_dir = os.path.join(os.path.dirname(__file__), "resources")

    torch.manual_seed(42)

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

    input_set = [
        (
            os.path.abspath(os.path.join(input_dir, "000009.jpg")),
            os.path.abspath(os.path.join(input_dir, "000009.txt")),
        ),
        (
            os.path.abspath(os.path.join(input_dir, "000013.jpg")),
            os.path.abspath(os.path.join(input_dir, "000013.txt")),
        ),
        (
            os.path.abspath(os.path.join(input_dir, "000022.jpg")),
            os.path.abspath(os.path.join(input_dir, "000022.txt")),
        ),
    ]
    input_image_path, calib_path = input_set[0]
    # ========================================================
    # OFT model configuration based on real model parameters
    logger.info(f"Generating Torch ref model")

    # 1 Handle inputs
    input_tensor = load_image(input_image_path, pad_hw=(H_PADDED, W_PADDED), dtype=torch.float32)[None].to(
        torch.float32
    )
    calib = load_calib(calib_path, dtype=torch.float32)[None].to(torch.float32)
    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=torch.float32)[None].to(
        torch.float32
    )

    # 2 Create reference OFTnet
    topdown_layers = 8
    ref_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        dtype=torch.float32,
    )

    ref_model = load_checkpoint(ref_model, model_location_generator)
    state_dict = create_OFT_model_parameters(ref_model, (input_tensor, calib, grid), device=device)
    # Apply model optimizations
    model_opt = ModelOptimizations()
    model_opt.apply(state_dict)

    # 3 Create reference encoder
    ref_encoder = ObjectEncoder(nms_thresh=NMS_THRESH, dtype=torch.float32)

    # 4 Run torch oftnet inference pass
    intermediates, scores, pos_offsets, dim_offsets, ang_offsets = ref_model(
        input_tensor, calib, grid, return_intermediates=False
    )

    # 5 Load encoder parameters
    scores = scores.squeeze(0)
    pos_offsets = pos_offsets.squeeze(0)
    dim_offsets = dim_offsets.squeeze(0)
    ang_offsets = ang_offsets.squeeze(0)
    grid_ = grid.clone().squeeze(0)
    decoder_params = create_decoder_model_parameters(
        ref_encoder, [scores, pos_offsets, dim_offsets, ang_offsets, grid_], device
    )

    # # 6 Run torch encoder inference pass
    # ref_outs, ref_enc_intermediates = ref_encoder.decode(scores, pos_offsets, dim_offsets, ang_offsets, grid_)
    # ref_objects = ref_encoder.create_objects(*ref_outs)

    if not args.torch:
        # ========================================================
        # TT model configuration
        logger.info(f"Generating TT model")

        # Create tt OFTnet
        tt_model = TTOftNet(
            device,
            state_dict,
            state_dict.layer_args,
            TTBasicBlock,
            [2, 2, 2, 2],
            ref_model.mean,
            ref_model.std,
            input_shape_hw=input_tensor.shape[2:],
            calib=calib,
            grid=grid,
            topdown_layers=topdown_layers,
            grid_res=GRID_RES,
            grid_height=GRID_HEIGHT,
        )

        # Create tt encoder
        tt_encoder = TTObjectEncoder(device, decoder_params, nms_thresh=NMS_THRESH)

    iter = 0
    sum_elapsed_time = sum_elapsed_time_1 = 0
    num_iter = args.num_iter if args.num_iter >= 0 else float("inf")
    cv2_thread.start()
    while True:
        input_image_path, calib_path = input_set[iter % len(input_set)]
        # ========================================================
        # Run ttnn inference pass
        logger.info(f"{iter+1}th run")
        start_time = time.time()

        input_tensor = load_image(input_image_path, pad_hw=(H_PADDED, W_PADDED), dtype=torch.float32)[None].to(
            torch.float32
        )
        calib = load_calib(calib_path, dtype=torch.float32)[None].to(torch.float32)
        grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=torch.float32)[None].to(
            torch.float32
        )

        # 1 Handle inputs
        tt_input = input_tensor.permute((0, 2, 3, 1))
        tt_input = ttnn.from_torch(tt_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tt_calib = ttnn.from_torch(calib, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_grid = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_grid_ = ttnn.from_torch(grid_, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # 2 Run tt oftnet inference pass
        (tt_intermediates, layer_names), tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = tt_model.forward(
            device, tt_input, tt_calib, tt_grid, return_intermediates=False
        )
        # _, _, _, _ = tt_scores.cpu(), tt_pos_offsets.cpu(), tt_dim_offsets.cpu(), tt_ang_offsets.cpu()  # ensure all ops are done
        # inference_time = time.time() - start_time
        # logger.info(f"TT OFTNet inference time: {inference_time:.4f} seconds")

        # 5 Run tt encoder inference pass
        tt_scores = ttnn.to_layout(ttnn.squeeze(tt_scores, 0), layout=ttnn.TILE_LAYOUT)
        tt_pos_offsets = ttnn.to_layout(ttnn.squeeze(tt_pos_offsets, 0), layout=ttnn.TILE_LAYOUT)
        tt_dim_offsets = ttnn.to_layout(ttnn.squeeze(tt_dim_offsets, 0), layout=ttnn.TILE_LAYOUT)
        tt_ang_offsets = ttnn.to_layout(ttnn.squeeze(tt_ang_offsets, 0), layout=ttnn.TILE_LAYOUT)

        tt_outs, tt_enc_intermediates, enc_names, enc_names_intermediates = tt_encoder.decode(
            device, tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets, tt_grid_
        )
        tt_objects = tt_encoder.create_objects(*tt_outs)
        e2e_time = time.time() - start_time
        sum_elapsed_time += e2e_time
        if iter > 0:
            sum_elapsed_time_1 += e2e_time
        print(f"TT OFTNet E2E time: {e2e_time:.4f} seconds, ({(1 / e2e_time):.2f} FPS)")
        # End of model inference
        # ========================================================
        for x in list(tt_intermediates) + [tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets]:
            if isinstance(x, ttnn.Tensor):
                ttnn.deallocate(x)

        # Visualize predictions
        input_tensor = input_tensor.to(torch.float32).squeeze(0)
        calib = calib.to(torch.float32).squeeze(0)
        img_out = visualize_objects_cv(input_tensor, calib, tt_objects)
        iter += 1
        with lock:
            shared_dict["output_image"] = img_out
        if not shared_dict["running"] or iter >= num_iter:
            shared_dict["running"] = False
            img_out = cv2.putText(img_out, "Press q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            with lock:
                shared_dict["output_image"] = img_out
            break
    cv2_thread.join()
    print(f"Total {iter} images processed.")
    print(f"[AVG] time taken = {sum_elapsed_time/iter:.4f} s")
    print(f"[AVG] FPS = {(iter/sum_elapsed_time):.2f} Hz")
    if iter > 1:
        print(f"[AVG-1] time taken = {sum_elapsed_time_1/(iter-1):.4f} s")
        print(f"[AVG-1] FPS = {((iter-1)/sum_elapsed_time_1):.2f} Hz")


if __name__ == "__main__":
    args = parse_args()
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    from conftest import model_location_generator

    main(device, model_location_generator, args)
    ttnn.close_device(device)
