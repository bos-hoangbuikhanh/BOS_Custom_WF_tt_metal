import argparse
import multiprocessing
import os
import time
from datetime import datetime

# import matplotlib
# matplotlib.use("QtAgg")
import cv2
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor

import ttnn
from models.bos_model.oft.demo.visualization.bbox import visualize_objects_cv
from models.bos_model.oft.demo.visualization.encoder import ObjectEncoder
from models.bos_model.oft.demo.visualization.kitti import KittiObjectDataset
from models.bos_model.oft.reference.oftnet import OftNet
from models.bos_model.oft.ttnn.model_preprocessing import create_OFT_model_parameters, prepare_ttnn_input, preprocessing
from models.bos_model.oft.ttnn.ttnn_oftnet import TtOftNet
from models.bos_model.oft.ttnn.ttnn_resnet import TtBasicBlock


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", "--num_images", type=int, default=10, help="number of images to process. -1 is infinite loop"
    )
    parser.add_argument(
        "-i",
        "--image_size",
        choices=[180, 256, 384],
        type=int,
        default=384,
        help="image height processed by bos model, width will be adjusted accordingly",
    )
    parser.add_argument("--torch", default=False, action="store_true", help="run torch model, not ttnn model")
    # parser.add_argument("--trace", default=False, action="store_true", help="enable trace mode")  # not implemented
    parser.add_argument("--save_result", default=False, action="store_true", help="save result images")

    args, _ = parser.parse_known_args(argv)
    return args


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


def main(device, model_type, res, args):
    # device.enable_program_cache()
    # ttnn.device.EnablePersistentKernelCache()

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

    class ModelSpec:
        model_path = "models/bos_model/oft/reference/checkpoint-0600.pth"
        grid_size = (80.0, 80.0)
        yoffset = 1.74
        nms_thresh = 0.2
        grid_height = 4.0
        grid_res = 0.5
        frontend = "resnet18"
        topdown = 8

    # Prepare dataset items once for pytest parameterization
    model_spec = ModelSpec()
    data_path = "models/bos_model/oft/data/kitti"
    dataset = KittiObjectDataset(data_path, "test", model_spec.grid_size, model_spec.grid_res, model_spec.yoffset)
    dataset_items = list(dataset)  # [(idx, image, calib, objects, grid), ...]

    # Build and load Torch model
    torch_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4.0,
    )
    ckpt = torch.load(model_spec.model_path, map_location="cpu")
    torch_model.load_state_dict(ckpt["model"])

    encoder = ObjectEncoder(nms_thresh=model_spec.nms_thresh)

    # --- Create TT model once ---
    if model_type == "Tt_model":
        # Dummy inputs to create parameters
        dummy_image = torch.zeros(1, 3, *res)
        dummy_calib = torch.eye(3, 4).unsqueeze(0)
        dummy_grid = torch.zeros(1, *dataset_items[0][4].shape)

        parameters = create_OFT_model_parameters(torch_model, (dummy_image, dummy_calib, dummy_grid), device=device)
        tt_module = TtOftNet(device, parameters, parameters.conv_args, TtBasicBlock, [2, 2, 2, 2])

    cv2_thread.start()
    cum_idx = 1
    elapsed_times = []
    if args.num_images < 0:  # -1
        args.num_images = float("inf")
    while True:
        img_idx, image, calib, objects, grid = dataset_items[(cum_idx - 1) % len(dataset_items)]
        image, calib, objects, grid = image.copy(), calib.clone(), objects, grid.clone()
        image = to_tensor(image)
        calib = adjust_calibration(calib, orig_size=(370, 1224), new_size=res)
        image = transform_image(image, res=res)

        if model_type == "Torch_model":
            start_time = time.time()
            pred_encoded = torch_model(image[None], calib[None], grid[None])
            elapsed_time = time.time() - start_time
            elapsed_times.append(elapsed_time)
            print(f"[{cum_idx}] ({img_idx}) Time taken = {elapsed_time:.4f} s", flush=True)
            print(f"[{cum_idx}] ({img_idx}) FPS = {(1 / (elapsed_time)):.2f} Hz", flush=True)
        else:
            # TT model path
            image = image.unsqueeze(0)
            calib = calib.unsqueeze(0)
            grid = grid.unsqueeze(0)

            pre_config = preprocessing(calib, grid, grid_height=4.0, cell_size=0.5, res=res, device=device)
            ttnn_input = prepare_ttnn_input(image, device=device)

            start_time = time.time()
            tt_pred_encoded = tt_module(device, ttnn_input, pre_config)
            elapsed_time = time.time() - start_time
            elapsed_times.append(elapsed_time)

            print(f"[{cum_idx}] ({img_idx}) Time taken = {elapsed_time:.4f} s", flush=True)
            print(f"[{cum_idx}] ({img_idx}) FPS = {(1 / (elapsed_time)):.2f} Hz", flush=True)

            pred_encoded = [ttnn.to_torch(x, dtype=torch.float32) for x in tt_pred_encoded]
            _ = [ttnn.deallocate(x) for x in tt_pred_encoded]

            grid = grid.squeeze(0)
            image = image.squeeze(0)
            calib = calib.squeeze(0)

        post_img = postprocess_and_save_results(
            model_type,
            image=image,
            calib=calib,
            grid=grid,
            pred_encoded=pred_encoded,
            objects=objects,
            encoder=encoder,
            res=res,
            output_base="models/bos_model/oft/demo/runs" if args.save_result else "",
        )

        with lock:
            shared_dict["output_image"] = post_img  # update shared_dict for cv_worker
        if not shared_dict["running"] or cum_idx >= args.num_images:
            shared_dict["running"] = False
            break
        cum_idx += 1
    cv2_thread.join()
    print(f"Total {cum_idx} images processed.")
    print(f"[AVG] time taken = {sum(elapsed_times)/len(elapsed_times):.4f} s")
    print(f"[AVG] FPS = {(len(elapsed_times)/sum(elapsed_times)):.2f} Hz")

    if len(elapsed_times) > 1:
        print(f"[AVG-1] time taken = {sum(elapsed_times[1:])/len(elapsed_times[1:]):.4f} s")
        print(f"[AVG-1] FPS = {(len(elapsed_times[1:])/sum(elapsed_times[1:])):.2f} Hz")


if __name__ == "__main__":
    args = parse_args()
    device = ttnn.open_device(device_id=0, l1_small_size=32768)

    model_type = "Torch_model" if args.torch else "Tt_model"
    image_res_map = {180: (180, 540), 256: (256, 832), 384: (384, 1248)}
    image_res = image_res_map.get(args.image_size, (384, 1248))
    main(device, model_type, image_res, args)
    ttnn.close_device(device)
