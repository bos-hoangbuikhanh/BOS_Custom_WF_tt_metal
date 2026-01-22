import copy
import importlib
import math
import os
from time import time

import supervision as sv
import torch
import torch.nn as nn
from ultralytics.engine.results import Results
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.block import DFL
from ultralytics.utils import ops
from ultralytics.utils.tal import dist2bbox, make_anchors

import ttnn
from models.bos_model.yolov8s.modules.YoloV8FPN import BOS_TTNN_Concat, BOS_TTNN_Upsample
from models.bos_model.yolov8s.modules.YoloV8Head import BOS_TTNN_Detect
from models.bos_model.yolov8s.modules.YoloV8Nets import BOS_TTNN_CSP, BOS_TTNN_SPPF, BOS_TTNN_Conv
from models.bos_model.yolov8s.reference.reference_model import YOLOv8_AlignedWithPT as Torch_YOLO
from models.bos_model.yolov8s.utilities.utility_functions import (
    _nearest_32,
    comp_pcc,
    divide,
    load_resize_and_pad_channels,
    setup_l1_sharded_input,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

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


class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc=80, ch=()):
        """Initialize the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.tensor([8, 16, 32])  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    # nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    # nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    Conv(x, c3, 3),
                    Conv(c3, c3, 3),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x, training=False):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        # return x

        if training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (List[torch.Tensor]): Input feature maps from different levels.

        Returns:
            (dict | tuple):

                - If in training mode, returns a dictionary containing outputs of both one2many and one2one detections.
                - If not in training mode, returns processed detections or a tuple with processed detections and raw outputs.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def post_detect_decode_bboxes(self, bboxes, anchors, xywh=True):
        return dist2bbox(bboxes, anchors, xywh, dim=1)

    def post_detect_inference(self, x, output_channels=80, reg_max=16, stride=[8, 16, 32]):
        # Inference path
        no = output_channels + reg_max * 4
        stride = torch.tensor(stride)
        # dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], no, -1) for xi in x], 2)
        anchors, strides = (x.transpose(0, 1) for x in make_anchors(x, stride, 0.5))
        anchors = anchors.unsqueeze(0)

        box, cls = x_cat.split((reg_max * 4, output_channels), 1)

        dfl_box = self.dfl(box)
        dbox = self.post_detect_decode_bboxes(dfl_box, anchors) * strides

        return torch.cat((dbox, cls.sigmoid()), 1), x

    def _inference(self, x):
        """
        Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (List[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and not (self.end2end or self.xyxy), dim=1)

    # @minyi : copied from ultralytics/models/yolo/detect/predict.py::DetectionPredictor::postprocess
    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        # breakpoint()
        save_feats = getattr(self, "save_feats", False)  # False
        preds = ops.non_max_suppression(
            preds,
            0.25,  # self.args.conf,
            0.7,  # self.args.iou,
            None,  # self.args.classes,
            False,  # self.args.agnostic_nms,
            max_det=self.max_det,
            nc=0,  # 0 if self.args.task == "detect" else len(self.model.names),
            end2end=False,  # getattr(self.model, "end2end", False),
            rotated=False,  # self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # if save_feats:
        #     obj_feats = self.get_obj_feats(self._feats, preds[1])
        #     preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        # if save_feats:
        #     for r, f in zip(results, obj_feats):
        #         r.feats = f  # add object features to results

        return results

    def get_obj_feats(self, feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min([x.shape[1] for x in feat_maps])  # find smallest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if len(idx) else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs):
        """
        Construct a list of Results objects from model predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.

        Returns:
            (List[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, ["image0.jpg"])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=class_names, boxes=pred[:, :6])


class YoloV8(nn.Module):
    def __init__(
        self,
        device=None,
        image_shape=[320, 320],
        in_channels=3,
        num_classes=80,
        #  depth=[1, 2, 2], width=[3, 32, 64, 128, 256, 512],
        depth=[1, 2, 2],
        width=[32, 32, 64, 128, 256, 512],
        use_checkpoint=True,
        layer_configs={},
        base_address="model.",
    ):
        super().__init__()

        self.device = device
        self.image_shape = image_shape
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.depth = depth
        self.width = width
        self.base_address = base_address

        self.input_tensor = None
        self.output0 = None
        self.output1 = None
        self.output2 = None

        self.detect_image_shapes = [
            divide(self.image_shape, 8),
            divide(self.image_shape, 16),
            divide(self.image_shape, 32),
        ]

        self.model = nn.ModuleList(
            [
                BOS_TTNN_Conv(
                    self.image_shape,
                    self.width[0],
                    self.width[1],
                    self.base_address + "0.",
                    self.device,
                    3,
                    2,
                    1,
                    batchnorm=True,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Conv(
                    divide(self.image_shape, 2),
                    self.width[1],
                    self.width[2],
                    self.base_address + "1.",
                    self.device,
                    3,
                    2,
                    1,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_CSP(
                    divide(self.image_shape, 4),
                    self.width[2],
                    self.width[2],
                    self.depth[0],
                    self.base_address + "2.",
                    shortcut=True,
                    device=self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Conv(
                    divide(self.image_shape, 4),
                    self.width[2],
                    self.width[3],
                    self.base_address + "3.",
                    self.device,
                    3,
                    2,
                    1,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_CSP(
                    divide(self.image_shape, 8),
                    self.width[3],
                    self.width[3],
                    self.depth[1],
                    self.base_address + "4.",
                    shortcut=True,
                    device=self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.975,
                ),
                BOS_TTNN_Conv(
                    divide(self.image_shape, 8),
                    self.width[3],
                    self.width[4],
                    self.base_address + "5.",
                    self.device,
                    3,
                    2,
                    1,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_CSP(
                    divide(self.image_shape, 16),
                    self.width[4],
                    self.width[4],
                    self.depth[1],
                    self.base_address + "6.",
                    shortcut=True,
                    device=self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.98,
                ),
                BOS_TTNN_Conv(
                    divide(self.image_shape, 16),
                    self.width[4],
                    self.width[5],
                    self.base_address + "7.",
                    self.device,
                    3,
                    2,
                    1,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_CSP(
                    divide(self.image_shape, 32),
                    self.width[5],
                    self.width[5],
                    self.depth[0],
                    self.base_address + "8.",
                    shortcut=True,
                    device=self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.98,
                ),
                BOS_TTNN_SPPF(
                    divide(self.image_shape, 32),
                    self.width[5],
                    self.width[5],
                    5,
                    1,
                    self.base_address + "9.",
                    self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Upsample(
                    divide(image_shape, 16),
                    self.width[5],
                    self.base_address + "10.",
                    self.device,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Concat(
                    -1,
                    self.base_address + "11",
                    self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_CSP(
                    divide(self.image_shape, 16),
                    self.width[-1] + self.width[-2],
                    self.width[-2],
                    1,
                    self.base_address + "12.",
                    self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Upsample(
                    divide(image_shape, 8),
                    self.width[4],
                    self.base_address + "13.",
                    self.device,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Concat(
                    -1,
                    self.base_address + "14",
                    self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.995
                ),
                BOS_TTNN_CSP(
                    divide(self.image_shape, 8),
                    self.width[-2] + self.width[-3],
                    self.width[-3],
                    1,
                    self.base_address + "15.",
                    self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Conv(
                    divide(self.image_shape, 8),
                    self.width[-3],
                    self.width[-3],
                    self.base_address + "16.",
                    self.device,
                    3,
                    2,
                    1,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Concat(
                    -1,
                    self.base_address + "17",
                    self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.995
                ),
                BOS_TTNN_CSP(
                    divide(self.image_shape, 16),
                    self.width[-2] + self.width[-3],
                    self.width[-2],
                    1,
                    self.base_address + "18.",
                    self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Conv(
                    divide(self.image_shape, 16),
                    self.width[-2],
                    self.width[-2],
                    self.base_address + "19.",
                    self.device,
                    3,
                    2,
                    1,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Concat(
                    -1,
                    self.base_address + "20",
                    self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.995
                ),
                BOS_TTNN_CSP(
                    divide(self.image_shape, 32),
                    self.width[-1] + self.width[-2],
                    self.width[-1],
                    1,
                    self.base_address + "21.",
                    self.device,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Detect(
                    self.detect_image_shapes,
                    width[-3:],
                    self.out_channels,
                    self.base_address + "22.",
                    device,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
            ]
        )

    def forward(self, x=None):
        x = x if x is not None else self.input_tensor
        m = self.model

        # Backbone
        x = m[0](x)
        x = m[1](x)
        x = m[2](x)
        x = m[3](x)
        x = m[4](x)
        p3 = x

        x = m[5](x)
        x = m[6](x)
        p4 = x

        x = m[7](x)
        x = m[8](x)

        x = m[9](x)

        p5 = x

        p5 = ttnn.sharded_to_interleaved(p5, ttnn.L1_MEMORY_CONFIG)
        p4 = ttnn.sharded_to_interleaved(p4, ttnn.L1_MEMORY_CONFIG)
        p3 = ttnn.sharded_to_interleaved(p3, ttnn.L1_MEMORY_CONFIG)

        # FPN
        x = m[10](p5)

        x = m[11](x, p4)
        p4_td = m[12](x, pad_if_optimal=True)

        p4_td = ttnn.sharded_to_interleaved(p4_td, ttnn.L1_MEMORY_CONFIG)
        x = m[13](p4_td)
        x = m[14](x, p3)
        p3_td = m[15](x)

        # PAN
        p3_down = m[16](p3_td)

        # core_grid, shard_shape = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 1))}), [32, 128]
        # reshard_shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        # reshard_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, reshard_shard_spec)
        # p3_down = ttnn.reshard(p3_down, reshard_mem_config)
        p3_down = ttnn.sharded_to_interleaved(p3_down, ttnn.L1_MEMORY_CONFIG)

        x = m[17](p3_down, p4_td, pad_if_optimal=True)
        p4_out = m[18](x, pad_if_optimal=True)
        p4_down = m[19](p4_out)

        # core_grid, shard_shape = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 1))}), [32, 256]
        # reshard_shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        # reshard_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, reshard_shard_spec)
        # p4_down = ttnn.reshard(p4_down, reshard_mem_config)
        p4_down = ttnn.sharded_to_interleaved(p4_down, ttnn.L1_MEMORY_CONFIG)

        p5_out = m[20](p4_down, p5, pad_if_optimal=True)
        p5_out = m[21](p5_out, pad_if_optimal=True)

        # self.output0 = p3_td
        # self.output1 = p4_out
        # self.output2 = p5_out

        # self.output0 = p3_td
        # self.output1 = p4_out
        # self.output2 = p5_out

        # Detect
        self.output0, self.output1, self.output2 = m[22]([p3_td, p4_out, p5_out])

        return [self.output0, self.output1, self.output2]


class YoloV8sRunner:
    def __init__(
        self,
        device_id,
        **kwargs,
    ):
        image_shape = kwargs.get("image_shape", [320, 320])
        self.input_channels = kwargs.get("input_channels", 3)
        self.output_classes = kwargs.get("output_classes", 80)
        self.trace = kwargs.get("trace", False)
        self.dataset = kwargs.get("dataset", "models/bos_model/yolov8s/reference/images/")
        weights_dir = kwargs.get("weights_dir", "models/bos_model/yolov8s")
        enable_persistent_cache = kwargs.get("enable_persistent_cache", True)

        self.model_name = "YoloV8s"
        self.image_height, self.image_width = (
            image_shape if isinstance(image_shape, list) else (image_shape, image_shape)
        )

        if enable_persistent_cache:
            ttnn.device.EnablePersistentKernelCache()

        if not self.trace:
            self.device = ttnn.open_device(device_id=device_id, l1_small_size=20480)
        else:
            self.device = ttnn.open_device(device_id=device_id, l1_small_size=20480, trace_region_size=10419200)

        self.test_images = [f for f in os.listdir(self.dataset) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        current_dir = os.path.dirname(os.path.abspath(__file__))
        layer_config_file = f"models.bos_model.yolov8s.configs.yolov8s_{self.image_height}x{self.image_width}"
        if not os.path.exists(os.path.join(current_dir, f"configs/{layer_config_file.split('.')[-1]}.py")):
            layer_config_file = f"models.bos_model.yolov8s.configs.yolov8s_256x256"
            print(f"Using {layer_config_file} as config file")
        layer_configs = importlib.import_module(layer_config_file).layer_configs

        self.bos_model = YoloV8(
            device=self.device,
            image_shape=(self.image_height, self.image_width),
            in_channels=self.input_channels,
            num_classes=self.output_classes,
            layer_configs=layer_configs,
        )
        self.bos_model.eval()

        self.torch_model = Torch_YOLO()
        self.torch_model.load_ckpt(os.path.join(weights_dir, "yolov8s.pt"))
        self.torch_model.eval()

        if self.trace:  # Warm up the model
            torch_image, original_size, input_size = load_resize_and_pad_channels(
                os.path.join(self.dataset, self.test_images[0]),
                image_size=(self.image_height, self.image_width),
                golden=True,
            )
            test_image_host, input_mem_config = setup_l1_sharded_input(self.device, torch_image)
            self.bos_model.input_tensor = test_image_host.to(self.device, input_mem_config)
            spec = self.bos_model.input_tensor.spec

            # Warm up
            outputs = self.bos_model()

            # Capture
            self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            outputs = self.bos_model()
            ttnn.end_trace_capture(self.device, self.tid, cq_id=0)

        self.golden_outputs = {}
        self.model_outputs = {}
        self.torch_outputs = {}
        self.performance = {}

        self.deallocated = False
        self.closed = False
        self.num_images = 0

    def run_golden(self):
        assert self.num_images > 0, "Please run the TTNN model first"
        images = self.test_images[: self.num_images]
        for i, image in enumerate(images):
            # print(f"Processing image {i+1}/{num_images}: {os.path.join(self.dataset, image)}")
            torch_image, original_size, input_size = load_resize_and_pad_channels(
                os.path.join(self.dataset, image), image_size=(self.image_height, self.image_width), golden=True
            )
            torch_outputs, _ = self.torch_model(torch_image)  # returns ([detect_22], [p3_td, p4_out, p5_out])
            self.golden_outputs[os.path.join(self.dataset, image)] = torch_outputs

        return self.golden_outputs

    def run_inference(self, num_images=3):
        self.num_images = num_images
        images = self.test_images[: self.num_images]
        for i, image in enumerate(images):
            torch_image, original_size, input_size = load_resize_and_pad_channels(
                os.path.join(self.dataset, image), image_size=(self.image_height, self.image_width), golden=True
            )
            start_time = time()
            if self.trace:
                test_image_host, input_mem_config = setup_l1_sharded_input(self.device, torch_image)
                ttnn.copy_host_to_device_tensor(test_image_host, self.bos_model.input_tensor, 0)
            else:
                test_image, input_mem_config = setup_l1_sharded_input(self.device, torch_image)
                test_image = test_image.to(self.device, input_mem_config)

            if self.trace:
                ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=True)
            else:
                outputs = self.bos_model(test_image)
            output0 = ttnn.from_device(self.bos_model.output0, blocking=True)
            output1 = ttnn.from_device(self.bos_model.output1, blocking=True)
            output2 = ttnn.from_device(self.bos_model.output2, blocking=True)
            elapsed_time = time() - start_time

            self.model_outputs[os.path.join(self.dataset, image)] = [output0, output1, output2]
            self.performance[os.path.join(self.dataset, image)] = elapsed_time

        return self.model_outputs

    def get_torch_outputs(self):
        if not self.golden_outputs:
            self.run_golden()
        if not self.model_outputs:
            self.run_inference()
        for key in self.model_outputs.keys():
            outputs = []
            for i, output in enumerate(self.model_outputs[key]):
                torch_output = tt_to_torch_tensor(output)
                torch_output = torch_output.permute(0, 3, 1, 2)
                torch_output = torch_output.reshape(self.golden_outputs[key][i].shape).to(torch.float32)
                outputs.append(torch_output)
            self.torch_outputs[key] = outputs
        return self.torch_outputs

    def check_pcc(self, tolerance=0.995, print_pcc=False, print_performance=False):
        if not self.golden_outputs:
            self.run_golden()
        if not self.torch_outputs:
            self.get_torch_outputs()
        for key in self.torch_outputs.keys():
            print(f"Checking image: {key}") if print_pcc else None
            for torch_output, golden_output in zip(self.torch_outputs[key], self.golden_outputs[key]):
                _, pcc = comp_pcc(golden_output, torch_output)
                print(f"\tPCC = {pcc:.3f}") if print_pcc else None
                if pcc < tolerance:
                    print(f"PCC check failed for image {key}: PCC = {pcc:.3f}")
            if print_performance:
                print(f"Inference time: {self.performance[key]*1000:.3f} ms ({1/self.performance[key]:.3f} FPS)")

    def check_performace(self):
        if not self.performance:
            self.run_inference()
        for key in self.performance.keys():
            print(
                f"Image: {key}, Inference time: {self.performance[key]*1000:.3f} ms ({1/self.performance[key]:.3f} FPS)"
            )

    def deallocate(self):
        for key in self.model_outputs.keys():
            for output in self.model_outputs[key]:
                ttnn.deallocate(output)
        self.deallocated = True

    def deallocate_and_close_device(self):
        if not self.deallocated:
            self.deallocate()
        ttnn.close_device(self.device)
        self.closed = True

    def aging_test(self, num_images=10000):
        pass

    def one_shot(self, num_images=3):
        self.run_inference(num_images=num_images)
        self.run_golden()
        self.check_pcc(print_pcc=True)
        self.check_performace()
        self.deallocate_and_close_device()

        return {
            "model_outputs": self.model_outputs,
            "golden_outputs": self.golden_outputs,
            # "Torch_Outputs": self.torch_outputs,
            "performance": self.performance,
        }


def run_e2e_per(device_id, batch_size=1, num_iter=3, **kwargs):
    fps = None
    output_tensor = None
    return {
        "fps": fps,
    }


def run_single_inference(device_id, torch_tensor, **kwargs):
    output_tensor = None
    return {
        "output_tensor": output_tensor,
    }
