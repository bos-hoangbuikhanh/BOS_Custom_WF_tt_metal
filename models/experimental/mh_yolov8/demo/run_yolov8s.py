import os, sys
from datetime import datetime
import math

import cv2
import pytest
import torch
from loguru import logger
from ultralytics import YOLO
from time import time
import argparse
import numpy as np
import supervision as sv
from ultralytics.utils import ops

import ttnn
from models.bos_model.mh_yolov8.demo.demo_utils import LoadImages, postprocess, preprocess
from models.bos_model.mh_yolov8.tt.ttnn_yolov8s import TtYolov8sModel
from models.bos_model.mh_yolov8.tt.tt_yolov8s_utils import custom_preprocessor
from models.common.utility_functions import disable_persistent_kernel_cache

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
    62: "TV",
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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.6, text_padding=1)


def save_yolo_predictions_by_model(result, model_save_dir, image_path, model_name):
    os.makedirs(model_save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if model_name == "torch_model":
        bounding_box_color, label_color = (0, 255, 0), (0, 255, 0)
    else:
        bounding_box_color, label_color = (255, 0, 0), (255, 255, 0)

    boxes = result["boxes"]["xyxy"]
    scores = result["boxes"]["conf"]
    classes = result["boxes"]["cls"]
    names = result["names"]

    if np.isnan(boxes).any() or np.isnan(scores).any() or np.isnan(classes).any():
        logger.error(f"NaN detected in result for {image_path}")
        return

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {score.item():.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), bounding_box_color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_name = os.path.basename(image_path)
    output_name = f"prediction_{image_name}"
    output_path = os.path.join(model_save_dir, output_name)

    cv2.imwrite(output_path, image)

    print(f"✅ Predictions saved to {output_path}")


def save_yolo_predictions_by_model_supervision(result, model_save_dir, image_path, input_size):
    os.makedirs(model_save_dir, exist_ok=True)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    for _, det in enumerate(result):
        if det is not None and len(det):
            boxes = det[:, :4].detach().cpu().numpy()
            confs = det[:, 4].detach().cpu().numpy()
            class_ids = det[:, 5].detach().cpu().numpy().astype(int)

            orig_h, orig_w, _ = original_image.shape
            input_h, input_w = input_size
            scale_x, scale_y = orig_w / input_w, orig_h / input_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            detections = sv.Detections(xyxy=boxes, confidence=confs, class_id=class_ids)
            labels = [f"{class_names.get(cls, str(cls))} {conf:.2f}" for cls, conf in zip(class_ids, confs)]

            annotated_image = box_annotator.annotate(scene=original_image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            filename = os.path.basename(image_path)
            out_path = os.path.join(model_save_dir, f"prediction_{filename}")
            cv2.imwrite(out_path, annotated_bgr)
            print(f"✅ Saved result: {out_path}\n")
        else:
            print(f"No detections found for image {image_path} <- len(det)={len(det)}\n")


def torch_pcc(x, y):
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return [torch_pcc(x[i], y[i]) for i in range(len(x))]
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"

    x_mean = x.mean()
    y_mean = y.mean()
    x_diff = x - x_mean
    y_diff = y - y_mean
    numerator = torch.sum(x_diff * y_diff)
    denominator = torch.sqrt(torch.sum(x_diff**2)) * torch.sqrt(torch.sum(y_diff**2))
    pcc = numerator / (denominator + 1e-8)
    return pcc.item()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num_images", type=int, default=100, help="number of images to process")
    parser.add_argument("--trace", default=False, action="store_true", help="enable trace mode")
    parser.add_argument("--cq2", default=False, action="store_true", help="enable 2cq mode")
    parser.add_argument("--torch", default=False, action="store_true", help="run torch model")
    parser.add_argument(
        "--save_pt",
        default=False,
        action="store_true",
        help="save intermediate tensors (not supported in trace mode yet)",
    )
    parser.add_argument(
        "--regen_input",
        default=False,
        action="store_true",
        help="regenerate every each layers for pcc debugging (not supported in trace mode yet)",
    )

    args = parser.parse_args()

    if args.cq2 and not args.trace:
        raise ValueError("2cq mode support only when trace mode enabled.")
    return args


def main(device, args):
    disable_persistent_kernel_cache()

    batch_size = 1
    image_height = 320
    image_width = 320
    input_channels = 3
    output_classes = len(class_names)  # 80
    model_type = "torch_model" if args.torch else "tt_model"

    images = [
        "models/bos_model/mh_yolov8/demo/images/16004479832_a748d55f21_k.jpg",
        "models/bos_model/mh_yolov8/demo/images/17790319373_bd19b24cfc_k.jpg",
        "models/bos_model/mh_yolov8/demo/images/18124840932_e42b3e377c_k.jpg",
        "models/bos_model/mh_yolov8/demo/images/19064748793_bb942deea1_k.jpg",
        "models/bos_model/mh_yolov8/demo/images/24274813513_0cfd2ce6d0_k.jpg",
        "models/bos_model/mh_yolov8/demo/images/33823288584_1d21cf0a26_k.jpg",
        "models/bos_model/mh_yolov8/demo/images/33887522274_eebd074106_k.jpg",
        "models/bos_model/mh_yolov8/demo/images/bus.jpg",
    ]
    images = images[: args.num_images]  # Limit number of images to process
    dataset = LoadImages(path=images)
    logger.info(f"Inferencing {len(dataset)} images with {model_type} model")

    save_dir = "models/bos_model/mh_yolov8/demo/runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(save_dir, f"{model_type}_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    # input_save_dir = os.path.join(save_dir, f"input_{timestamp}")  # debugging for pre-processing
    # os.makedirs(input_save_dir, exist_ok=True)

    torch_model = YOLO("yolov8s.pt").model
    torch_model.eval()

    if model_type == "tt_model":
        if args.trace:
            if args.cq2:
                from models.bos_model.mh_yolov8.tests.yolov8s_e2e_performant import Yolov8sTrace2CQ

                yolov8s_trace_2cq = Yolov8sTrace2CQ()
                logger.info(f"Warming up trace_2cq model")
                yolov8s_trace_2cq.initialize_yolov8s_trace_2cqs_inference(
                    device,
                    batch_size,
                    (image_height, image_width),
                )
            else:  # trace only
                from models.bos_model.mh_yolov8.tests.yolov8s_e2e_performant import Yolov8sTrace

                yolov8s_trace = Yolov8sTrace()
                logger.info(f"Warming up trace model")
                yolov8s_trace.initialize_yolov8s_trace_inference(
                    device,
                    batch_size,
                    (image_height, image_width),
                )
        else:
            state_dict = torch_model.state_dict()
            parameters = custom_preprocessor(device, state_dict, inp_h=image_height, inp_w=image_width)
            model = TtYolov8sModel(device=device, parameters=parameters, res=(image_height, image_width))
            logger.info("Inferencing using ttnn Model")

    elapsed_times = []
    pcc_all = []
    for batch in dataset:
        paths, im0s, s = batch

        im = preprocess(im0s, res=(image_height, image_width))
        # debugging for pre-processing
        # im_t = im.squeeze(0).permute((1, 2, 0)) * 255  #  CHW -> HWC
        # im_t = im_t.cpu().numpy()
        # output_path = os.path.join(input_save_dir, f"input_{os.path.basename(paths[0])}")
        # cv2.imwrite(output_path, im_t)

        if model_type == "torch_model":
            model = torch_model
            if args.save_pt:
                torch.save(im, "torch_im.pt")
                conv_0 = model.model[0](im)
                torch.save(conv_0, "torch_conv_0.pt")
                conv_1 = model.model[1](conv_0)
                torch.save(conv_1, "torch_conv_1.pt")

                c2f_2 = model.model[2](conv_1)
                torch.save(c2f_2, "torch_c2f_2.pt")
                c2f_2_cv1 = model.model[2].cv1(conv_1)
                torch.save(c2f_2_cv1, "torch_c2f_2.cv1.pt")
                c2f_2_y = list(c2f_2_cv1.chunk(2, 1))
                torch.save(c2f_2_y[0], "torch_c2f_2.cv1_a.pt")
                torch.save(c2f_2_y[1], "torch_c2f_2.cv1_b.pt")
                # c2f_2_y.extend(m(c2f_2_y[-1]) for m in model.model[2].m)
                for i, m in enumerate(model.model[2].m):
                    m_cv1 = m.cv1(c2f_2_y[-1])
                    torch.save(m_cv1, f"torch_c2f_2.m{i}.cv1.pt")
                    m_cv2 = m.cv2(m_cv1)
                    torch.save(m_cv2, f"torch_c2f_2.m{i}.cv2.pt")
                    if m.add:
                        m_cv2 = m_cv2 + c2f_2_y[-1]
                    torch.save(m_cv2, f"torch_c2f_2.m{i}.add.pt")
                    c2f_2_y.extend([m_cv2])
                c2f_2_y = torch.cat(c2f_2_y, 1)
                torch.save(c2f_2_y, "torch_c2f_2.concat.pt")
                c2f_2_cv2 = model.model[2].cv2(c2f_2_y)
                torch.save(c2f_2_cv2, "torch_c2f_2.cv2.pt")  # same to "torch_c2f_2.pt"
                assert torch_pcc(c2f_2_cv2, c2f_2) > 0.999, breakpoint()

                conv_3 = model.model[3](c2f_2)
                torch.save(conv_3, "torch_conv_3.pt")

                c2f_4 = model.model[4](conv_3)
                torch.save(c2f_4, "torch_c2f_4.pt")
                c2f_4_cv1 = model.model[4].cv1(conv_3)
                torch.save(c2f_4_cv1, "torch_c2f_4.cv1.pt")
                c2f_4_y = list(c2f_4_cv1.chunk(2, 1))
                torch.save(c2f_4_y[0], "torch_c2f_4.cv1_a.pt")
                torch.save(c2f_4_y[1], "torch_c2f_4.cv1_b.pt")
                for i, m in enumerate(model.model[4].m):
                    m_cv1 = m.cv1(c2f_4_y[-1])
                    torch.save(m_cv1, f"torch_c2f_4.m{i}.cv1.pt")
                    m_cv2 = m.cv2(m_cv1)
                    torch.save(m_cv2, f"torch_c2f_4.m{i}.cv2.pt")
                    if m.add:
                        m_cv2 = m_cv2 + c2f_4_y[-1]
                    torch.save(m_cv2, f"torch_c2f_4.m{i}.add.pt")
                    c2f_4_y.extend([m_cv2])
                c2f_4_y = torch.cat(c2f_4_y, 1)
                torch.save(c2f_4_y, "torch_c2f_4.concat.pt")
                c2f_4_cv2 = model.model[4].cv2(c2f_4_y)
                torch.save(c2f_4_cv2, "torch_c2f_4.cv2.pt")  # same to "torch_c2f_4.pt"
                assert torch_pcc(c2f_4_cv2, c2f_4) > 0.999, breakpoint()

                conv_5 = model.model[5](c2f_4)
                torch.save(conv_5, "torch_conv_5.pt")

                c2f_6 = model.model[6](conv_5)  # [1, 256, 40, 40]
                torch.save(c2f_6, "torch_c2f_6.pt")
                c2f_6_cv1 = model.model[6].cv1(conv_5)
                torch.save(c2f_6_cv1, "torch_c2f_6.cv1.pt")
                c2f_6_y = list(c2f_6_cv1.chunk(2, 1))
                torch.save(c2f_6_y[0], "torch_c2f_6.cv1_a.pt")
                torch.save(c2f_6_y[1], "torch_c2f_6.cv1_b.pt")
                for i, m in enumerate(model.model[6].m):
                    m_cv1 = m.cv1(c2f_6_y[-1])
                    torch.save(m_cv1, f"torch_c2f_6.m{i}.cv1.pt")
                    m_cv2 = m.cv2(m_cv1)
                    torch.save(m_cv2, f"torch_c2f_6.m{i}.cv2.pt")
                    if m.add:
                        m_cv2 = m_cv2 + c2f_6_y[-1]
                    torch.save(m_cv2, f"torch_c2f_6.m{i}.add.pt")
                    c2f_6_y.extend([m_cv2])
                c2f_6_y = torch.cat(c2f_6_y, 1)
                torch.save(c2f_6_y, "torch_c2f_6.concat.pt")
                c2f_6_cv2 = model.model[6].cv2(c2f_6_y)
                torch.save(c2f_6_cv2, "torch_c2f_6.cv2.pt")  # same to "torch_c2f_6.pt"
                assert torch_pcc(c2f_6_cv2, c2f_6) > 0.999, breakpoint()

                conv_7 = model.model[7](c2f_6)
                torch.save(conv_7, "torch_conv_7.pt")

                c2f_8 = model.model[8](conv_7)
                torch.save(c2f_8, "torch_c2f_8.pt")
                c2f_8_cv1 = model.model[8].cv1(conv_7)
                torch.save(c2f_8_cv1, "torch_c2f_8.cv1.pt")
                c2f_8_y = list(c2f_8_cv1.chunk(2, 1))
                torch.save(c2f_8_y[0], "torch_c2f_8.cv1_a.pt")
                torch.save(c2f_8_y[1], "torch_c2f_8.cv1_b.pt")
                for i, m in enumerate(model.model[8].m):
                    m_cv1 = m.cv1(c2f_8_y[-1])
                    torch.save(m_cv1, f"torch_c2f_8.m{i}.cv1.pt")
                    m_cv2 = m.cv2(m_cv1)
                    torch.save(m_cv2, f"torch_c2f_8.m{i}.cv2.pt")
                    if m.add:
                        m_cv2 = m_cv2 + c2f_8_y[-1]
                    torch.save(m_cv2, f"torch_c2f_8.m{i}.add.pt")
                    c2f_8_y.extend([m_cv2])
                c2f_8_y = torch.cat(c2f_8_y, 1)
                torch.save(c2f_8_y, "torch_c2f_8.concat.pt")
                c2f_8_cv2 = model.model[8].cv2(c2f_8_y)
                torch.save(c2f_8_cv2, "torch_c2f_8.cv2.pt")  # same to "torch_c2f_8.pt"
                assert torch_pcc(c2f_8_cv2, c2f_8) > 0.999, breakpoint()

                sppf_9 = model.model[9](c2f_8)
                torch.save(sppf_9, "torch_sppf_9.pt")
                sppf_9_cv1 = model.model[9].cv1(c2f_8)
                torch.save(sppf_9_cv1, "torch_sppf_9.cv1.pt")
                sppf_9_y = [sppf_9_cv1]
                for i in range(3):
                    sppf_9_m = model.model[9].m(sppf_9_y[-1])
                    torch.save(sppf_9_m, f"torch_sppf_9.maxpool{i}.pt")
                    sppf_9_y.append(sppf_9_m)
                sppf_9_y = torch.cat(sppf_9_y, 1)
                torch.save(sppf_9_y, "torch_sppf_9.concat.pt")
                sppf_9_cv2 = model.model[9].cv2(sppf_9_y)
                torch.save(sppf_9_cv2, "torch_sppf_9.cv2.pt")
                assert torch_pcc(sppf_9_cv2, sppf_9) > 0.999, breakpoint()

                upsample_10 = model.model[10](sppf_9)  # [1, 512, 40, 40]
                torch.save(upsample_10, "torch_upsample_10.pt")
                concat_11 = model.model[11]((upsample_10, c2f_6))  # [1, 768, 40, 40]
                torch.save(concat_11, "torch_concat_11.pt")

                c2f_12 = model.model[12](concat_11)
                torch.save(c2f_12, "torch_c2f_12.pt")
                c2f_12_cv1 = model.model[12].cv1(concat_11)
                torch.save(c2f_12_cv1, "torch_c2f_12.cv1.pt")
                c2f_12_y = list(c2f_12_cv1.chunk(2, 1))
                torch.save(c2f_12_y[0], "torch_c2f_12.cv1_a.pt")
                torch.save(c2f_12_y[1], "torch_c2f_12.cv1_b.pt")
                for i, m in enumerate(model.model[12].m):
                    m_cv1 = m.cv1(c2f_12_y[-1])
                    torch.save(m_cv1, f"torch_c2f_12.m{i}.cv1.pt")
                    m_cv2 = m.cv2(m_cv1)
                    torch.save(m_cv2, f"torch_c2f_12.m{i}.cv2.pt")
                    if m.add:
                        m_cv2 = m_cv2 + c2f_12_y[-1]
                    torch.save(m_cv2, f"torch_c2f_12.m{i}.add.pt")
                    c2f_12_y.extend([m_cv2])
                c2f_12_y = torch.cat(c2f_12_y, 1)
                torch.save(c2f_12_y, "torch_c2f_12.concat.pt")
                c2f_12_cv2 = model.model[12].cv2(c2f_12_y)
                torch.save(c2f_12_cv2, "torch_c2f_12.cv2.pt")  # same to "torch_c2f_12.pt"
                assert torch_pcc(c2f_12_cv2, c2f_12) > 0.999, breakpoint()

                upsample_13 = model.model[13](c2f_12)
                torch.save(upsample_13, "torch_upsample_13.pt")
                concat_14 = model.model[14]((upsample_13, c2f_4))
                torch.save(concat_14, "torch_concat_14.pt")

                c2f_15 = model.model[15](concat_14)
                torch.save(c2f_15, "torch_c2f_15.pt")
                c2f_15_cv1 = model.model[15].cv1(concat_14)
                torch.save(c2f_15_cv1, "torch_c2f_15.cv1.pt")
                c2f_15_y = list(c2f_15_cv1.chunk(2, 1))
                torch.save(c2f_15_y[0], "torch_c2f_15.cv1_a.pt")
                torch.save(c2f_15_y[1], "torch_c2f_15.cv1_b.pt")
                for i, m in enumerate(model.model[15].m):
                    m_cv1 = m.cv1(c2f_15_y[-1])
                    torch.save(m_cv1, f"torch_c2f_15.m{i}.cv1.pt")
                    m_cv2 = m.cv2(m_cv1)
                    torch.save(m_cv2, f"torch_c2f_15.m{i}.cv2.pt")
                    if m.add:
                        m_cv2 = m_cv2 + c2f_15_y[-1]
                    torch.save(m_cv2, f"torch_c2f_15.m{i}.add.pt")
                    c2f_15_y.extend([m_cv2])
                c2f_15_y = torch.cat(c2f_15_y, 1)
                torch.save(c2f_15_y, "torch_c2f_15.concat.pt")
                c2f_15_cv2 = model.model[15].cv2(c2f_15_y)
                torch.save(c2f_15_cv2, "torch_c2f_15.cv2.pt")  # same to "torch_c2f_15.pt"
                assert torch_pcc(c2f_15_cv2, c2f_15) > 0.999, breakpoint()

                conv_16 = model.model[16](c2f_15)
                torch.save(conv_16, "torch_conv_16.pt")
                concat_17 = model.model[17]((conv_16, c2f_12))
                torch.save(concat_17, "torch_concat_17.pt")

                c2f_18 = model.model[18](concat_17)
                torch.save(c2f_18, "torch_c2f_18.pt")
                c2f_18_cv1 = model.model[18].cv1(concat_17)
                torch.save(c2f_18_cv1, "torch_c2f_18.cv1.pt")
                c2f_18_y = list(c2f_18_cv1.chunk(2, 1))
                torch.save(c2f_18_y[0], "torch_c2f_18.cv1_a.pt")
                torch.save(c2f_18_y[1], "torch_c2f_18.cv1_b.pt")
                for i, m in enumerate(model.model[18].m):
                    m_cv1 = m.cv1(c2f_18_y[-1])
                    torch.save(m_cv1, f"torch_c2f_18.m{i}.cv1.pt")
                    m_cv2 = m.cv2(m_cv1)
                    torch.save(m_cv2, f"torch_c2f_18.m{i}.cv2.pt")
                    if m.add:
                        m_cv2 = m_cv2 + c2f_18_y[-1]
                    torch.save(m_cv2, f"torch_c2f_18.m{i}.add.pt")
                    c2f_18_y.extend([m_cv2])
                c2f_18_y = torch.cat(c2f_18_y, 1)
                torch.save(c2f_18_y, "torch_c2f_18.concat.pt")
                c2f_18_cv2 = model.model[18].cv2(c2f_18_y)
                torch.save(c2f_18_cv2, "torch_c2f_18.cv2.pt")  # same to "torch_c2f_18.pt"
                assert torch_pcc(c2f_18_cv2, c2f_18) > 0.999, breakpoint()

                conv_19 = model.model[19](c2f_18)
                torch.save(conv_19, "torch_conv_19.pt")
                concat_20 = model.model[20]((conv_19, sppf_9))
                torch.save(concat_20, "torch_concat_20.pt")

                c2f_21 = model.model[21](concat_20)
                torch.save(c2f_21, "torch_c2f_21.pt")
                c2f_21_cv1 = model.model[21].cv1(concat_20)
                torch.save(c2f_21_cv1, "torch_c2f_21.cv1.pt")
                c2f_21_y = list(c2f_21_cv1.chunk(2, 1))
                torch.save(c2f_21_y[0], "torch_c2f_21.cv1_a.pt")
                torch.save(c2f_21_y[1], "torch_c2f_21.cv1_b.pt")
                for i, m in enumerate(model.model[21].m):
                    m_cv1 = m.cv1(c2f_21_y[-1])
                    torch.save(m_cv1, f"torch_c2f_21.m{i}.cv1.pt")
                    m_cv2 = m.cv2(m_cv1)
                    torch.save(m_cv2, f"torch_c2f_21.m{i}.cv2.pt")
                    if m.add:
                        m_cv2 = m_cv2 + c2f_21_y[-1]
                    torch.save(m_cv2, f"torch_c2f_21.m{i}.add.pt")
                    c2f_21_y.extend([m_cv2])
                c2f_21_y = torch.cat(c2f_21_y, 1)
                torch.save(c2f_21_y, "torch_c2f_21.concat.pt")
                c2f_21_cv2 = model.model[21].cv2(c2f_21_y)
                torch.save(c2f_21_cv2, "torch_c2f_21.cv2.pt")  # same to "torch_c2f_21.pt"
                assert torch_pcc(c2f_21_cv2, c2f_21) > 0.999, breakpoint()

                torch.save([c2f_15, c2f_18, c2f_21], "torch_detect_22.in.pt")
                detect_22 = model.model[22]([c2f_15, c2f_18, c2f_21])
                torch.save(detect_22[0], "torch_detect_22.out0.pt")
                torch.save(detect_22[1], "torch_detect_22.out1.pt")

                detect_22_conv_out = []
                for i, x in enumerate([c2f_15, c2f_18, c2f_21]):
                    a = model.model[22].cv2[i](x)
                    torch.save(a, f"torch_detect_22.{i}.cv2.pt")
                    a_0 = model.model[22].cv2[i][0](x)
                    a_1 = model.model[22].cv2[i][1](a_0)
                    a_2 = model.model[22].cv2[i][2](a_1)
                    torch.save(a_0, f"torch_detect_22.{i}.cv2.0.pt")
                    torch.save(a_1, f"torch_detect_22.{i}.cv2.1.pt")
                    torch.save(a_2, f"torch_detect_22.{i}.cv2.2.pt")
                    assert torch_pcc(a_2, a) > 0.999, breakpoint()

                    b = model.model[22].cv3[i](x)
                    torch.save(b, f"torch_detect_22.{i}.cv3.pt")
                    b_0 = model.model[22].cv3[i][0](x)
                    b_1 = model.model[22].cv3[i][1](b_0)
                    b_2 = model.model[22].cv3[i][2](b_1)
                    torch.save(b_0, f"torch_detect_22.{i}.cv3.0.pt")
                    torch.save(b_1, f"torch_detect_22.{i}.cv3.1.pt")
                    torch.save(b_2, f"torch_detect_22.{i}.cv3.2.pt")
                    assert torch_pcc(b_2, b) > 0.999, breakpoint()

                    cat = torch.cat((a, b), dim=1)
                    torch.save(cat, f"torch_detect_22.{i}.concat.pt")
                    detect_22_conv_out.append(cat)
                detect_22_infer = model.model[22]._inference(detect_22_conv_out)
                assert torch_pcc(detect_22_infer, detect_22[0]) > 0.999, breakpoint()
                torch.save(detect_22_infer, "torch_detect_22.pt")
                torch.save(detect_22_infer[:, :4, :], "torch_detect_22.box.pt")
                torch.save(detect_22_infer[:, 4:, :], "torch_detect_22.cls.pt")

            output0, output1 = model(im)
            if args.save_pt:
                torch.save(output0, f"torch_model_output.0.pt")
                assert torch_pcc(output0, detect_22[0]) > 0.999, breakpoint()
                for i, out in enumerate(output1):  # c2f_15, c2f_18, c2f_21
                    torch.save(out, f"torch_model_output.1.{i}.pt")
                    assert torch_pcc(out, detect_22[1][i]) > 0.999, breakpoint()
        else:  # ttnn
            if args.trace:
                if args.cq2:
                    logger.info("Execute trace + 2cqs model inference")
                    start_time = time()
                    (
                        tt_inputs_host,
                        sharded_mem_config_DRAM,
                        input_mem_config,
                    ) = yolov8s_trace_2cq.test_infra.setup_dram_sharded_input(device, im)
                    output0, output1 = yolov8s_trace_2cq.execute_yolov8s_trace_2cqs_inference(tt_inputs_host)
                    output0 = ttnn.from_device(output0, blocking=True)
                    elapsed_time = time() - start_time
                else:  # trace only
                    logger.info("Execute trace model inference")
                    start_time = time()
                    tt_inputs_host, input_mem_config = yolov8s_trace.test_infra.setup_l1_sharded_input(device, im)
                    output0, output1 = yolov8s_trace.execute_yolov8s_trace_inference(tt_inputs_host)
                    output0 = ttnn.from_device(output0, blocking=True)
                    elapsed_time = time() - start_time
                torch_output0 = ttnn.to_torch(output0, dtype=torch.float32)
            else:
                logger.info("Execute simple model inference")
                ttnn_im = im.permute((0, 2, 3, 1))  # NCHW -> NHWC
                n, h, w, c = ttnn_im.shape
                min_channels = 16
                start_time = time()
                if c < min_channels:
                    channel_padding_needed = min_channels - c
                    ttnn_im = torch.nn.functional.pad(ttnn_im, (0, channel_padding_needed, 0, 0, 0, 0), value=0.0)
                    c = min_channels
                ttnn_im = ttnn_im.reshape(1, 1, n * h * w, c)
                input_mem_config = ttnn.create_sharded_memory_config(
                    [n, h, w, c],
                    ttnn.CoreGrid(x=5, y=4),
                    ttnn.ShardStrategy.HEIGHT,
                )
                ttnn_input = ttnn.from_torch(ttnn_im, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                ttnn_input = ttnn_input.to(device, input_mem_config)

                output0, output1 = model(x=ttnn_input, _save_pt=args.save_pt, _regen_input=args.regen_input)
                output0 = ttnn.from_device(output0, blocking=True)
                elapsed_time = time() - start_time
                torch_output0 = ttnn.to_torch(output0, dtype=torch.float32)
                ttnn.deallocate(output0)
                _ = [ttnn.deallocate(x) for x in output1 if x.is_allocated()]
                if ttnn_input.is_allocated() and ttnn_input.devices():
                    ttnn.deallocate(ttnn_input)

            elapsed_times.append(elapsed_time)
            print(f"{paths} elapsed_time =", elapsed_time * 1000, "ms")
            print(f"{paths} FPS =", 1 / (elapsed_time), "Hz")

            torch_golden_output0, torch_golden_output1 = torch_model(im)
            pcc = torch_pcc(torch_output0, torch_golden_output0)
            pcc_all.append(pcc)
            print(f"{paths} PCC =", pcc)

        # results = postprocess(output0, im, im0s, batch, class_names)[0]
        # save_yolo_predictions_by_model(results, model_save_dir, paths[0], model_type)
        results = ops.non_max_suppression(torch_output0, conf_thres=0.25, iou_thres=0.45)
        save_yolo_predictions_by_model_supervision(results, model_save_dir, paths[0], (image_height, image_width))

    logger.info("Inference done")
    if model_type == "tt_model":
        fps_all = [1 / x for x in elapsed_times]
        print(f"pcc_all =", pcc_all)
        print(f"elapsed_times =", elapsed_times)
        print(f"fps_all =", fps_all)
        print(f"{len(elapsed_times)} images avgerage FPS =", len(elapsed_times) / sum(elapsed_times), "Hz")
        print(f"{len(elapsed_times)}-1 images avgerage FPS =", len(elapsed_times[1:]) / sum(elapsed_times[1:]), "Hz")


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
def test_trace_2cq_main(
    device,
    use_program_cache,
    model_location_generator,
):
    args = type(
        "args",
        (object,),
        {
            "trace": True,
            "cq2": True,
            "save_pt": False,
            "regen_input": False,
            "torch": False,
            "num_images": 100,
        },
    )
    main(device, args)


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816}], indirect=True)
def test_trace_main(
    device,
    use_program_cache,
    model_location_generator,
):
    args = type(
        "args",
        (object,),
        {
            "trace": True,
            "cq2": False,
            "save_pt": False,
            "regen_input": False,
            "torch": False,
            "num_images": 100,
        },
    )
    main(device, args)


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_main(
    device,
    use_program_cache,
    model_location_generator,
):
    args = type(
        "args",
        (object,),
        {
            "trace": False,
            "cq2": False,
            "save_pt": False,
            "regen_input": False,
            "torch": False,
            "num_images": 100,
        },
    )
    main(device, args)


if __name__ == "__main__":
    args = parse_args()
    device_id = 0
    if args.trace:
        if args.cq2:
            device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=3686400, num_command_queues=2)
        else:
            device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=3686400)
        ttnn.enable_program_cache(device)
    else:
        # device = ttnn.CreateDevice(device_id, l1_small_size=10240)
        device = ttnn.CreateDevice(device_id, l1_small_size=24576)
        ttnn.enable_program_cache(device)

    main(device, args)
    ttnn.close_device(device)
