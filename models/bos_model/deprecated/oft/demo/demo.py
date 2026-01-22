import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pytest
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor

import ttnn
from models.bos_model.oft.demo.visualization.bbox import visualize_objects
from models.bos_model.oft.demo.visualization.encoder import ObjectEncoder
from models.bos_model.oft.demo.visualization.kitti import KittiObjectDataset
from models.bos_model.oft.reference.oftnet import OftNet
from models.bos_model.oft.ttnn.model_preprocessing import create_OFT_model_parameters, prepare_ttnn_input, preprocessing
from models.bos_model.oft.ttnn.ttnn_oftnet import TtOftNet
from models.bos_model.oft.ttnn.ttnn_resnet import TtBasicBlock


def adjust_calibration(calib, orig_size, new_size=(180, 540)):
    scale_x = new_size[1] / orig_size[1]  # Width scale
    scale_y = new_size[0] / orig_size[0]  # Height scale
    calib[0, 0] *= scale_x  # fx (focal length x)
    calib[1, 1] *= scale_y  # fy (focal length y)
    calib[0, 2] *= scale_x  # cx (principal point x)
    calib[1, 2] *= scale_y  # cy (principal point y)
    return calib


def transform_image(image: torch.Tensor, res=(180, 540)) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize(res),
        ]
    )

    return transform(image)


def postprocess_and_save_results(
    model_type,
    image,
    calib,
    grid,
    pred_encoded,
    objects,
    encoder,
    output_base="models/bos_model/oft/demo/runs",
    res=(180, 540),
):
    # Decode predictions
    pred_encoded = [t[0].cpu() for t in pred_encoded]
    detections = encoder.decode(*pred_encoded, grid.cpu())

    # Plot predictions and ground truth
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 8))

    visualize_objects(image, calib, detections, ax=ax1)
    ax1.set_title(f"{model_type} Detections")

    visualize_objects(image, calib, objects, ax=ax2)
    ax2.set_title("Ground truth")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = os.path.join(output_base, model_type)
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"result_{timestamp}_res{res[0]}x{res[1]}.jpg")
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Predictions saved to: {save_path}")
    plt.pause(0.01)
    time.sleep(0.5)


class DummyArgs:
    model_path = "models/bos_model/oft/reference/checkpoint-0600.pth"
    root = "models/bos_model/oft/data/kitti"
    grid_size = (80.0, 80.0)
    yoffset = 1.74
    nms_thresh = 0.2
    grid_height = 4.0
    grid_res = 0.5
    frontend = "resnet18"
    topdown = 8
    num_test_images = 10  # default


# Prepare dataset items once for pytest parameterization
args = DummyArgs()
dataset = KittiObjectDataset(args.root, "test", args.grid_size, args.grid_res, args.yoffset)
dataset_items = list(dataset)[: args.num_test_images]  # [(idx, image, calib, objects, grid), ...]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "model_type",
    [
        ("Tt_model"),
        # ("Torch_model")  # Uncomment to test Torch model
    ],
)
@pytest.mark.parametrize(
    "res",
    [(384, 1248)],
)
def test_demo(device, model_type, res):
    device.enable_program_cache()
    args = DummyArgs()

    # Build and load Torch model
    torch_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4.0,
    )
    ckpt = torch.load(args.model_path, map_location="cpu")
    torch_model.load_state_dict(ckpt["model"])

    encoder = ObjectEncoder(nms_thresh=args.nms_thresh)

    # --- Create TT model once ---
    if model_type == "Tt_model":
        # Dummy inputs to create parameters
        dummy_image = torch.zeros(1, 3, *res)
        dummy_calib = torch.eye(3, 4).unsqueeze(0)
        dummy_grid = torch.zeros(1, *dataset_items[0][4].shape)

        parameters = create_OFT_model_parameters(torch_model, (dummy_image, dummy_calib, dummy_grid), device=device)
        tt_module = TtOftNet(device, parameters, parameters.conv_args, TtBasicBlock, [2, 2, 2, 2])

    elapsed_times = []

    for _, image, calib, objects, grid in dataset_items:
        image = to_tensor(image)
        calib = adjust_calibration(calib, orig_size=(370, 1224), new_size=res)
        image = transform_image(image, res=res)

        if model_type == "Torch_model":
            start_time = time.time()
            pred_encoded = torch_model(image[None], calib[None], grid[None])
            elapsed_time = time.time() - start_time
            elapsed_times.append(elapsed_time)
            print(f"[{_}] Time taken = {elapsed_time:.4f} s")
            print(f"[{_}] FPS = {(1 / (elapsed_time)):.2f} Hz")
        else:
            # TT model path
            image = image.unsqueeze(0)
            calib = calib.unsqueeze(0)
            grid = grid.unsqueeze(0)

            pre_config = preprocessing(calib, grid, grid_height=4.0, cell_size=0.5, res=res, device=device)
            ttnn_input = prepare_ttnn_input(image, device=device)

            start_time = time.time()
            pred_encoded = tt_module(device, ttnn_input, pre_config)
            elapsed_time = time.time() - start_time
            elapsed_times.append(elapsed_time)

            print(f"[{_}] Time taken = {elapsed_time:.4f} s")
            print(f"[{_}] FPS = {(1 / (elapsed_time)):.2f} Hz")

            pred_encoded = [ttnn.to_torch(x, dtype=torch.float32) for x in pred_encoded]

            grid = grid.squeeze(0)
            image = image.squeeze(0)
            calib = calib.squeeze(0)

        postprocess_and_save_results(
            model_type,
            image=image,
            calib=calib,
            grid=grid,
            pred_encoded=pred_encoded,
            objects=objects,
            encoder=encoder,
            res=res,
        )
    print(f"[AVG] time taken = {sum(elapsed_times)/len(elapsed_times):.4f} s")
    print(f"[AVG] FPS = {(len(elapsed_times)/sum(elapsed_times)):.2f} Hz")

    if len(elapsed_times) > 1:
        print(f"[AVG-1] time taken = {sum(elapsed_times[1:])/len(elapsed_times[1:]):.4f} s")
        print(f"[AVG-1] FPS = {(len(elapsed_times[1:])/sum(elapsed_times[1:])):.2f} Hz")
