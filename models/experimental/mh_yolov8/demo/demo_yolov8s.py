import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from ultralytics.utils import ops
from ultralytics.engine.results import Results
from ultralytics.nn.modules.block import DFL
from ultralytics.utils.tal import dist2bbox, make_anchors

# from reference.reference_model import Detect
# from run_yolov8s import Detect
import supervision as sv


import os
import copy
import torchvision.transforms as T
from PIL import Image
import cv2
import math

# import threading
# from multiprocessing import Process, Lock, Manager
import multiprocessing

import ttnn
from models.common.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    _nearest_32,
    _nearest_y,
)

import pytest

# from time import time
import time

# from reference.reference_model import YOLOv8_AlignedWithPT as Torch_YOLO
# from revamped_yolov8s import YoloV8 as YoloV8_revamped
from ultralytics import YOLO
from models.bos_model.mh_yolov8.tt.ttnn_yolov8s import TtYolov8sModel
from models.bos_model.mh_yolov8.tt.tt_yolov8s_utils import custom_preprocessor

from collections import deque

curr_file_path = os.path.dirname(os.path.realpath(__file__))

# Global shared variables and synchronization tools
image_height = 320
image_width = 320

class_names = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def resize_and_pad_channels(frame, image_size=[256, 256], golden=False):
    original_h, original_w, _ = frame.shape  # frame is a np array called by cv2
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(image_size),
        ]
    )
    tensor = transform(frame)

    c, h, w = tensor.shape
    if golden == False:
        if c % 32 != 0:
            pad_c = _nearest_32(c) - c
            pad_tensor = torch.zeros((pad_c, h, w), dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad_tensor], dim=0)

        tensor = tensor.unsqueeze(0).to(dtype=torch.bfloat16)
    else:
        tensor = tensor.unsqueeze(0)

    return tensor, (original_w, original_h), image_size


def resize_(frame, image_size=[256, 256], golden=False):
    original_h, original_w, _ = frame.shape  # frame is a np array called by cv2
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(image_size),
        ]
    )
    tensor = transform(frame)
    return tensor, (original_w, original_h), image_size


def pad_channels_(tensor, min_channels=8):
    c, h, w = tensor.shape
    if c % min_channels != 0:
        pad_c = _nearest_y(c, min_channels) - c
        pad_tensor = torch.zeros((pad_c, h, w), dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad_tensor], dim=0)

    tensor = tensor.unsqueeze(0).to(dtype=torch.bfloat16)
    return tensor


def setup_l1_sharded_input(device, torch_input_tensor=None, min_channels=8, num_cores=20):
    if num_cores == 20:
        core_grid = ttnn.CoreGrid(y=4, x=5)
    else:
        core_grid = ttnn.CoreGrid(y=8, x=8)
    torch_input_tensor = (
        torch.rand((1, min_channels, image_height, image_width), dtype=torch.float32)
        if torch_input_tensor is None
        else torch_input_tensor
    )
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
    shard_size = _nearest_y(nhw / num_cores, min_channels)
    input_mem_config = ttnn.create_sharded_memory_config(
        [1, 1, shard_size * num_cores, c],
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
    )
    # input_mem_config = ttnn.create_sharded_memory_config(
    #     [n, h, w, c],
    #     core_grid,
    #     ttnn.ShardStrategy.HEIGHT,
    # )
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return tt_inputs_host, input_mem_config


def video_worker(video_path, output_video_path, lock, shared_dict):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"video file not found: {video_path}")
        shared_dict["running"] = False
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if output_video_path:
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # set Supervision Annotator
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.6, text_padding=1)
    fps_overlay = True

    predict_result = shared_dict.copy()  # initial copy

    previous_time = time.time()
    # sleep_time = 0.04  # sec
    cv_wait_key = 1
    last_update_frame = 0
    current_frame = 0
    while shared_dict["running"]:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # infinite loop
            continue
        current_frame += 1

        with lock:
            # if current_frame % 10 == 0:
            # print(f"[1] update current_frame:{current_frame}")
            shared_dict["current_frame"] = current_frame
            shared_dict["shared_frame"] = frame.copy()
            if shared_dict["last_frame"] > last_update_frame:
                predict_result = shared_dict.copy()
                last_update_frame = shared_dict["last_frame"]
            accum_processed_frames = shared_dict["accum_processed_frames"]

        if predict_result["bboxes"] is not None:
            mask = predict_result["class_ids"] <= 13  # draw only class 1~13
            mask *= ~(
                (predict_result["class_ids"] == 9) * (predict_result["scores"] <= 0.3)
            )  # traffic_sign require conf_score over 0.3
            predict_result["bboxes"] = predict_result["bboxes"][mask]
            predict_result["scores"] = predict_result["scores"][mask]
            predict_result["class_ids"] = predict_result["class_ids"][mask]

            detections = sv.Detections(
                xyxy=predict_result["bboxes"], confidence=predict_result["scores"], class_id=predict_result["class_ids"]
            )
            labels = [
                f"{class_names.get(cls, str(cls))} {conf:.2f}"
                for cls, conf in zip(predict_result["class_ids"], predict_result["scores"])
            ]

            annotated_image = box_annotator.annotate(scene=frame, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            frame = annotated_image

        if fps_overlay:
            video_fps = 1 / (time.time() - previous_time)
            previous_time = time.time()
            # sleep_time = max(0, sleep_time + (1/fps - 1/video_fps)/100)
            cv2.putText(
                frame,
                f"frame[{shared_dict['current_frame']}] | fps: {video_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"model[{last_update_frame}]({accum_processed_frames}) | fps: {predict_result['fps']:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        if output_video_path:
            out_video.write(frame)
        if last_update_frame > 0:
            cv2.imshow("Annotated Video", frame)

        key = cv2.waitKey(cv_wait_key) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            cv_wait_key = int(not (cv_wait_key))
        elif key == ord("1"):
            fps_overlay = not (fps_overlay)

    shared_dict["running"] = False
    cap.release()
    if output_video_path:
        out_video.release()
    cv2.destroyAllWindows()

    pass


def main(device):
    video_path = os.path.join(curr_file_path, "videos/demo_logo_text_v1.1.mp4")
    output_video_path = ""
    # output_video_path = os.path.join(curr_file_path, 'results/annotated_demo_logo_text_v1.1.mp4')

    # Start inference thread
    lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    shared_dict = manager.dict(
        {
            "running": True,
            "current_frame": 0,
            "shared_frame": None,
            "last_frame": -1,
            "bboxes": None,
            "scores": None,
            "class_ids": None,
            "fps": 0,
            "accum_processed_frames": 0,
        }
    )

    video_thread = multiprocessing.Process(target=video_worker, args=(video_path, output_video_path, lock, shared_dict))

    # Load model
    torch_model = YOLO("yolov8s.pt").model
    torch_model.eval()

    state_dict = torch_model.state_dict()
    parameters = custom_preprocessor(device, state_dict, inp_h=image_height, inp_w=image_width)
    ttnn_model = TtYolov8sModel(device=device, parameters=parameters, res=(image_height, image_width))

    torch_run = False
    model = torch_model if torch_run else ttnn_model

    tid = 0
    if not torch_run:
        ttnn.enable_program_cache(device)
        # enable_persistent_kernel_cache()

        #    print("Warming up and Capturing the model for trace mode")
        test_image_host, input_mem_config = setup_l1_sharded_input(device)

        # Warm up
        model.input_tensor = test_image_host.to(device, input_mem_config)
        spec = model.input_tensor.spec
        _, _ = model()
        model.input_tensor = test_image_host.to(device, input_mem_config)
        ttnn.deallocate(model.output_tensor[0])
        _ = [ttnn.deallocate(x) for x in model.output_tensor[1]]

        # Capture
        model_input_tensor_addr = model.input_tensor.buffer_address()
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        _, _ = model()
        tt_device_tensor = ttnn.allocate_tensor_on_device(spec, device)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        print(
            f"model_input_tensor_addr: {model_input_tensor_addr}, tt_device_tensor: {tt_device_tensor.buffer_address()}"
        )
        assert model_input_tensor_addr == tt_device_tensor.buffer_address()

    video_thread.start()
    last_update_frame = 0
    accum_processed_frames = 0
    while shared_dict["running"]:
        with lock:
            current_frame = shared_dict["current_frame"]

        if last_update_frame < current_frame:
            #     print(f"{time.strftime('%X',time.localtime())} [inference] received {current_frame} frame")
            last_update_frame = current_frame
            frame_for_infer = shared_dict["shared_frame"].copy()
        else:
            continue

        if frame_for_infer is not None:
            # print(f"{time.strftime('%X',time.localtime())} [inference] run {current_frame} frame")
            if torch_run:
                start = time.time()
                img_tensor, original_size, input_size = resize_and_pad_channels(
                    frame_for_infer, image_size=[image_height, image_width], golden=True
                )
                output_y, output_x = model(img_tensor)
                end = time.time()
            else:  # ttnn
                img_tensor, original_size, input_size = resize_(frame_for_infer, image_size=[image_height, image_width])
                start = time.time()
                img_tensor = pad_channels_(img_tensor, min_channels=16)

                test_image_host, input_mem_config = setup_l1_sharded_input(device, img_tensor)
                ttnn.copy_host_to_device_tensor(test_image_host, tt_device_tensor, 0)

                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                output_y = ttnn.from_device(model.output_tensor[0], blocking=True)
                end = time.time()
                output_y = ttnn.to_torch(output_y, dtype=torch.float32)

            nms_result = ops.non_max_suppression(output_y, conf_thres=0.25, iou_thres=0.7)

            det = nms_result[0]
            if det is not None and len(det):
                boxes = det[:, :4].detach().cpu().numpy()
                confs = det[:, 4].detach().cpu().numpy()
                class_ids = det[:, 5].detach().cpu().numpy().astype(int)

                orig_w, orig_h = original_size
                input_w, input_h = input_size
                scale_x, scale_y = orig_w / input_w, orig_h / input_h
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

                with lock:
                    print(f"{time.strftime('%X',time.localtime())} [inference] update {current_frame} result")
                    shared_dict["last_frame"] = last_update_frame
                    shared_dict["bboxes"] = boxes.copy()
                    shared_dict["scores"] = confs.copy()
                    shared_dict["class_ids"] = class_ids.copy()
                    shared_dict["fps"] = 1 / (end - start)
                    shared_dict["accum_processed_frames"] = accum_processed_frames
            else:
                with lock:
                    # print(f"{time.strftime('%X',time.localtime())} [inference] faild to detect {current_frame}")
                    shared_dict["last_frame"] = last_update_frame
                    shared_dict["bboxes"] = None
                    shared_dict["scores"] = None
                    shared_dict["class_ids"] = None
                    shared_dict["fps"] = 1 / (end - start)
                    shared_dict["accum_processed_frames"] = accum_processed_frames
            accum_processed_frames += 1
            print(f"{time.strftime('%X',time.localtime())} [inference] accum_processed_frames {accum_processed_frames}")
            print(f"{time.strftime('%X',time.localtime())} [inference] fps {shared_dict['fps']}")
            print(f"{time.strftime('%X',time.localtime())} [inference] ========================")
    video_thread.join()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=20480, trace_region_size=10419200)
    # device = torch.device("cpu") # for torch
    main(device)
