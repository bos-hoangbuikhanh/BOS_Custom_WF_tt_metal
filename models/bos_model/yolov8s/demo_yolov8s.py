import importlib
import math
import multiprocessing
import os
import time

import cv2
import supervision as sv
import torch
from reference.reference_model import YOLOv8_AlignedWithPT as Torch_YOLO

# from reference.reference_model import Detect
from run_yolov8s import Detect
from ultralytics.utils import ops
from utilities.utility_functions import _nearest_32, pad_channels_, resize_, resize_and_pad_channels, tt_to_torch_tensor
from yolov8s import YoloV8

import ttnn

BUFFER_SIZE = 5

curr_file_path = os.path.dirname(os.path.realpath(__file__))

# Global shared variables and synchronization tools
# Currently we're not supporting changing the image height/width.
image_height = 256
image_width = 256

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


def setup_l1_sharded_input(device, torch_input_tensor=None, min_channels=32, num_cores=20):
    if num_cores == 20:
        core_grid = ttnn.CoreGrid(y=4, x=5)
    else:
        core_grid = ttnn.CoreGrid(y=8, x=8)
    if torch_input_tensor is None:  # torch input should be NHWC
        torch_input_tensor = torch.rand((1, 32, image_height, image_width), dtype=torch.float32)
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
    shard_size = _nearest_32(nhw / num_cores)
    input_mem_config = ttnn.create_sharded_memory_config(
        [1, 1, shard_size * num_cores, c],
        ttnn.CoreGrid(x=5, y=4),
        ttnn.ShardStrategy.HEIGHT,
    )
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
                xyxy=predict_result["bboxes"],
                confidence=predict_result["scores"],
                class_id=predict_result["class_ids"],
            )
            labels = [
                f"{class_names.get(cls, str(cls))} {conf:.2f}"
                for cls, conf in zip(predict_result["class_ids"], predict_result["scores"])
            ]

            # fake_frame / frame_buffer 사용 제거, 그냥 현재 frame 사용
            annotated_image = box_annotator.annotate(scene=frame, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            frame = annotated_image

        if fps_overlay:
            video_fps = 1 / (time.time() - previous_time)
            previous_time = time.time()
            cv2.putText(
                frame,
                # BUFFER_SIZE 빼지 말고 그냥 현재 frame 번호 표기
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


def main(device):
    video_path = os.path.join(curr_file_path, "videos/demo_logo_text_v1.1.mp4")
    output_video_path = ""
    # output_video_path = os.path.join(curr_file_path, 'results/annotated_demo_logo_text_v1.1.mp4')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"video file not found: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = None
    if output_video_path:
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Supervision annotator 설정 (원하면 사용)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.6, text_padding=1)
    fps_overlay = True

    layer_config_file = f"models.bos_model.yolov8s.configs.yolov8s_{image_height}x{image_width}"
    if not os.path.exists(os.path.join(curr_file_path, f"configs/{layer_config_file.split('.')[-1]}.py")):
        layer_config_file = f"models.bos_model.yolov8s.configs.yolov8s_256x256"
        print(f"Using {layer_config_file} as config file")
    layer_configs = importlib.import_module(layer_config_file).layer_configs

    bos_model = YoloV8(
        device=device,
        image_shape=[image_height, image_width],
        in_channels=3,
        num_classes=len(class_names),
        layer_configs=layer_configs,
    )
    bos_model.eval()

    torch_model = Torch_YOLO()
    torch_model.load_ckpt(os.path.join(curr_file_path, "yolov8s.pt"))
    torch_model.eval()

    torch_detect_layer = Detect(nc=len(class_names), ch=[128, 256, 512])
    torch_detect_weights = torch.load(os.path.join(curr_file_path, "detect_weights.pth"))
    torch_detect_layer.load_state_dict(torch_detect_weights)
    torch_detect_layer.eval()

    torch_run = False
    model = torch_model if torch_run else bos_model

    # ttnn trace capture 준비
    tid = 0
    if not torch_run:
        device.enable_program_cache()
        # ttnn.device.EnablePersistentKernelCache()

        test_image_host, input_mem_config = setup_l1_sharded_input(device)
        model.input_tensor = test_image_host.to(device, input_mem_config)

        # warmup
        _ = model()

        # capture
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        _ = model()
        ttnn.end_trace_capture(device, tid, cq_id=0)

    current_frame = 0
    accum_processed_frames = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            # 한 번만 돌고 끝낼 거면 break
            # 무한 루프 돌리고 싶으면 아래 주석을 해제
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            break

        current_frame += 1

        # ==== 전처리 ====
        if torch_run:
            start = time.time()
            img_tensor, original_size, input_size = resize_and_pad_channels(
                frame, image_size=[image_height, image_width], golden=True
            )
        else:
            img_tensor, original_size, input_size = resize_(frame, image_size=[image_height, image_width])
            start = time.time()
            img_tensor = pad_channels_(img_tensor)

            # host -> device copy
            test_image_host, input_mem_config = setup_l1_sharded_input(device, img_tensor)
            ttnn.copy_host_to_device_tensor(test_image_host, model.input_tensor, 0)

        # ==== 추론 ====
        if torch_run:
            outputs = model(img_tensor)
            end = time.time()
            final_output = outputs  # 필요에 맞게 수정
        else:
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            output0 = ttnn.from_device(model.output0, blocking=True)
            output1 = ttnn.from_device(model.output1, blocking=True)
            output2 = ttnn.from_device(model.output2, blocking=True)
            end = time.time()

            detect_heads = []
            for output in [output0, output1, output2]:
                torch_output = tt_to_torch_tensor(output)
                ttnn.deallocate(output)
                torch_output = torch_output.permute(0, 3, 1, 2)
                b, k, _, p = torch_output.shape
                p2 = int(math.sqrt(p))
                torch_output = torch_output.reshape([b, k, -1, p2]).to(torch.float32)
                detect_heads.append(torch_output)

            final_output = torch_detect_layer.post_detect_inference(detect_heads)

        # ==== NMS & bbox rescale ====
        nms_result = ops.non_max_suppression(final_output[0], conf_thres=0.25, iou_thres=0.7)
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

            # ========= ✨ 필터링 구간 추가 ✨ =========
            # class_ids <= 13
            # 그리고 traffic light(class_id == 9)는 conf > 0.3 조건 필요
            mask = (class_ids <= 13) & ~((class_ids == 9) & (confs <= 0.3))

            boxes = boxes[mask]
            confs = confs[mask]
            class_ids = class_ids[mask]

            detections = sv.Detections(
                xyxy=boxes,
                confidence=confs,
                class_id=class_ids,
            )
            labels = [f"{class_names.get(cls, str(cls))} {conf:.2f}" for cls, conf in zip(class_ids, confs)]

            frame_anno = box_annotator.annotate(scene=frame.copy(), detections=detections)
            frame_anno = label_annotator.annotate(scene=frame_anno, detections=detections, labels=labels)
        else:
            frame_anno = frame

        frame_fps = 1.0 / (time.time() - prev_time)
        prev_time = time.time()
        model_fps = 1.0 / (end - start)
        accum_processed_frames += 1

        if fps_overlay:
            cv2.putText(
                frame_anno,
                f"frame[{current_frame}] | fps: {frame_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame_anno,
                f"model({accum_processed_frames}) | fps: {model_fps:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        if out_video is not None:
            out_video.write(frame_anno)

        # 실시간으로 보고 싶으면:
        cv2.imshow("Annotated Video", frame_anno)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        print(f"{time.strftime('%X', time.localtime())} [inference] frame {current_frame}")
        print(f"{time.strftime('%X', time.localtime())} [inference] accum_processed_frames {accum_processed_frames}")
        print(f"{time.strftime('%X', time.localtime())} [inference] model fps {model_fps}")
        print(f"{time.strftime('%X', time.localtime())} [inference] ========================")

    cap.release()
    if out_video is not None:
        out_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # device = ttnn.open_device(device_id=0, l1_small_size=20480, trace_region_size=10419200)
    device_dict = {"device_id": 0, "l1_small_size": 20480 * 2, "trace_region_size": 10419200}
    ttnn_device = ttnn._ttnn.device
    device = ttnn_device.CreateDevice(**device_dict)
    main(device)
