# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from time import time

import cv2
import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.bos_model.mh_yolov8.demo.demo_utils import LoadImages, postprocess, preprocess
from models.bos_model.mh_yolov8.tt.tt_yolov8s_utils import custom_preprocessor
from models.bos_model.mh_yolov8.tt.ttnn_yolov8s import TtYolov8sModel
from models.common.utility_functions import disable_persistent_kernel_cache


def save_yolo_predictions_by_model(result, save_dir, image_path, model_name):
    model_save_dir = os.path.join(save_dir, model_name)
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

    logger.info(f"✅ Predictions saved to {output_path}")


def torch_fuse(conv, bn, eps=1e-03):
    bn_weight = bn.weight.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = bn.bias.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_mean = bn.running_mean.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = bn.running_var.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = conv.weight.clone()
    weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
    bias = bias.squeeze()

    return weight, bias


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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "source, model_type",
    [
        ("models/bos_model/mh_yolov8/demo/images/bus.jpg", "torch_model"),
        # ("models/bos_model/mh_yolov8/demo/images/16004479832_a748d55f21_k.jpg", "torch_model"),
        # ("models/bos_model/mh_yolov8/demo/images/17790319373_bd19b24cfc_k.jpg", "torch_model"),
        # ("models/bos_model/mh_yolov8/demo/images/18124840932_e42b3e377c_k.jpg", "torch_model"),
        # ("models/bos_model/mh_yolov8/demo/images/19064748793_bb942deea1_k.jpg", "torch_model"),
        # ("models/bos_model/mh_yolov8/demo/images/24274813513_0cfd2ce6d0_k.jpg", "torch_model"),
        # ("models/bos_model/mh_yolov8/demo/images/33823288584_1d21cf0a26_k.jpg", "torch_model"),
        # ("models/bos_model/mh_yolov8/demo/images/33887522274_eebd074106_k.jpg", "torch_model"),
        # (["models/bos_model/mh_yolov8/demo/images/bus.jpg", "models/bos_model/mh_yolov8/demo/images/16004479832_a748d55f21_k.jpg"], "torch_model"),  # program cache test
        ("models/bos_model/mh_yolov8/demo/images/bus.jpg", "tt_model"),
        # ("models/bos_model/mh_yolov8/demo/images/16004479832_a748d55f21_k.jpg", "tt_model"),
        # ("models/bos_model/mh_yolov8/demo/images/17790319373_bd19b24cfc_k.jpg", "tt_model"),
        # ("models/bos_model/mh_yolov8/demo/images/18124840932_e42b3e377c_k.jpg", "tt_model"),
        # ("models/bos_model/mh_yolov8/demo/images/19064748793_bb942deea1_k.jpg", "tt_model"),
        # ("models/bos_model/mh_yolov8/demo/images/24274813513_0cfd2ce6d0_k.jpg", "tt_model"),
        # ("models/bos_model/mh_yolov8/demo/images/33823288584_1d21cf0a26_k.jpg", "tt_model"),
        # ("models/bos_model/mh_yolov8/demo/images/33887522274_eebd074106_k.jpg", "tt_model"),
        # (["models/bos_model/mh_yolov8/demo/images/bus.jpg", "models/bos_model/mh_yolov8/demo/images/16004479832_a748d55f21_k.jpg"], "tt_model"),  # program cache test
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(320, 320)])
def test_demo(
    device, source, model_type, res, use_weights_from_ultralytics, use_program_cache, save_pt=False, regen_input=False
):
    disable_persistent_kernel_cache()

    if use_weights_from_ultralytics:
        torch_model = YOLO("yolov8s.pt")
        torch_model = torch_model.model
        model = torch_model.eval()

    if model_type == "tt_model":
        state_dict = torch_model.state_dict()
        parameters = custom_preprocessor(device, state_dict, inp_h=res[0], inp_w=res[1])
        model = TtYolov8sModel(device=device, parameters=parameters, res=(res[0], res[1]))
        logger.info("Inferencing using ttnn Model")

    save_dir = "models/bos_model/mh_yolov8/demo/runs/demo"

    dataset = LoadImages(path=source)

    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)

    names = {
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

    for batch in dataset:
        paths, im0s, s = batch
        logger.info(f"Processing {s[0]}")

        im = preprocess(im0s, res=res)

        if model_type == "torch_model":
            if save_pt:
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
                torch.save(detect_22[0], "torch_detect_22.y.pt")
                torch.save(detect_22[0][:, :4, :], "torch_detect_22.y.0.pt")
                torch.save(detect_22[0][:, 4:, :], "torch_detect_22.y.1.pt")
                # torch.save(detect_22[1], "torch_detect_22.x.pt")

                # detect_22 breakdown
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

                # _inference breakdown
                no, reg_max, nc = model.model[22].no, model.model[22].reg_max, model.model[22].nc
                assert nc == len(names)
                detect_22_cat = torch.cat([xi.view(1, no, -1) for xi in detect_22_conv_out], 2)
                torch.save(detect_22_cat, f"torch_detect_22._inference.concat.pt")
                detect_22_box, detect_22_cls = detect_22_cat.split((reg_max * 4, nc), 1)
                torch.save(detect_22_box, f"torch_detect_22._inference.box.pt")
                torch.save(detect_22_cls, f"torch_detect_22._inference.cls.pt")
                detect_22_dfl = model.model[22].dfl(detect_22_box)
                torch.save(detect_22_dfl, f"torch_detect_22._inference.dfl.pt")

                # dfl breakdown
                b, _, a = detect_22_box.shape
                c1 = model.model[22].dfl.c1
                detect_22_dfl_reshape = detect_22_box.view(b, 4, c1, a)
                torch.save(detect_22_dfl_reshape, f"torch_detect_22._inference.dfl.reshape.pt")
                detect_22_dfl_softmax = torch.nn.functional.softmax(detect_22_dfl_reshape, dim=2)
                torch.save(detect_22_dfl_softmax, f"torch_detect_22._inference.dfl.softmax.pt")
                detect_22_dfl_conv = model.model[22].dfl.conv(detect_22_dfl_softmax.transpose(2, 1))
                torch.save(detect_22_dfl_conv, f"torch_detect_22._inference.dfl.conv.pt")
                torch.save(
                    model.model[22].dfl.conv.weight, f"torch_detect_22._inference.dfl.conv.weight.pt"
                )  # [0, 1, 2, 3, ..., 15]
                detect_22_dfl_out = detect_22_dfl_conv.view(b, 4, a)
                torch.save(detect_22_dfl_out, f"torch_detect_22._inference.dfl.out.pt")
                assert torch_pcc(detect_22_dfl_out, detect_22_dfl) > 0.999, breakpoint()

                anchors = model.model[22].anchors.unsqueeze(0)
                strides = model.model[22].strides
                detect_22_dbox = model.model[22].decode_bboxes(detect_22_dfl, anchors)
                torch.save(detect_22_dbox, f"torch_detect_22._inference.dbox.pt")

                # decode_bboxes breakdown
                lt, rb = detect_22_dfl_out.split(2, dim=1)
                torch.save(lt, f"torch_detect_22._inference.dbox.lt.pt")
                torch.save(rb, f"torch_detect_22._inference.dbox.rb.pt")
                x1y1 = anchors - lt
                x2y2 = anchors + rb
                torch.save(x1y1, f"torch_detect_22._inference.dbox.x1y1.pt")
                torch.save(x2y2, f"torch_detect_22._inference.dbox.x2y2.pt")
                c_xy = (x1y1 + x2y2) / 2
                torch.save(c_xy, f"torch_detect_22._inference.dbox.c_xy.pt")
                wh = x2y2 - x1y1
                torch.save(wh, f"torch_detect_22._inference.dbox.wh.pt")
                detect_22_dbox_out = torch.cat((c_xy, wh), 1)  # xywh bbox
                assert torch_pcc(detect_22_dbox_out, detect_22_dbox) > 0.999, breakpoint()

                detect_22_dbox *= strides
                torch.save(detect_22_dbox, "torch_detect_22._inference.dbox_strides.pt")
                # detect_22_cls = detect_22_cls.sigmoid()
                detect_22_cls = torch.nn.functional.sigmoid(detect_22_cls)
                torch.save(detect_22_cls, "torch_detect_22._inference.cls_sigmoid.pt")

                detect_22_infer_out = torch.cat((detect_22_dbox, detect_22_cls), 1)
                assert torch_pcc(detect_22_infer_out, detect_22_infer) > 0.999, breakpoint()

            preds = model(im)
            if save_pt:
                for i, pred in enumerate(preds[0]):
                    torch.save(pred, f"torch_preds.{i}.pt")
                    assert torch_pcc(preds[i], detect_22[i]) > 0.999, breakpoint()
        else:  # ttnn
            """
            # pad input channels to 16 to avoid slow interleaved2sharded codepath for 3/8 channels
            ttnn_input = torch.nn.functional.pad(im, (0, 0, 0, 0, 0, 13, 0, 0), value=0)
            ttnn_input = ttnn_input.permute((0, 2, 3, 1))
            ttnn_input = ttnn_input.reshape(
                1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]
            )
            ttnn_input = ttnn.from_torch(im, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            """

            ttnn_im = im.permute((0, 2, 3, 1))  # NCHW -> NHWC
            n, h, w, c = ttnn_im.shape
            min_channels = 8
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

            start_time = time()
            preds = model(x=ttnn_input, _save_pt=save_pt, _regen_input=regen_input)
            preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32)
            elapsed_time = time() - start_time
            print(f"{source} elapsed_time =", elapsed_time * 1000, "ms")
            print(f"{source} FPS =", 1 / (elapsed_time), "Hz")

        results = postprocess(preds, im, im0s, batch, names)[0]

        save_yolo_predictions_by_model(results, save_dir, paths[0], model_type)

    logger.info("Inference done")
