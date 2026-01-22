import copy
import math

import torch
import torch.nn as nn
from ultralytics.engine.results import Results
from ultralytics.nn.modules import SPPF, C2f, Conv, DWConv
from ultralytics.nn.modules.block import DFL

# from reference_model import YOLOv8_AlignedWithPT as YOLO
from ultralytics.utils import ops
from ultralytics.utils.tal import dist2bbox, make_anchors

# from ultralytics.nn.modules.head import Detect


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
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc=80, ch=()):
        """Initialize the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        # self.stride = torch.zeros(self.nl)  # strides computed during build
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
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x, pre_inference=False):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if pre_inference:
            return x

        if self.training:  # Training path
            return x
        y = self._inference(x)
        # torch.save(x[0], "refer_x0.pt")
        # torch.save(x[1], "refer_x1.pt")
        # torch.save(x[2], "refer_x2.pt")
        # torch.save(y, "refer_y.pt")
        # breakpoint()
        return y if self.export else (y, x)

    # @minyi : I changed the postprocess func so that this will not work like what originally intended
    # def forward_end2end(self, x):
    #     """
    #     Performs forward pass of the v10Detect module.
    #
    #     Args:
    #         x (List[torch.Tensor]): Input feature maps from different levels.
    #
    #     Returns:
    #         (dict | tuple):
    #
    #             - If in training mode, returns a dictionary containing outputs of both one2many and one2one detections.
    #             - If not in training mode, returns processed detections or a tuple with processed detections and raw outputs.
    #     """
    #     x_detach = [xi.detach() for xi in x]
    #     one2one = [
    #         torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
    #     ]
    #     for i in range(self.nl):
    #         x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    #     if self.training:  # Training path
    #         return {"one2many": x, "one2one": one2one}
    #
    #     y = self._inference(one2one)
    #     y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
    #     return y if self.export else (y, {"one2many": x, "one2one": one2one})

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


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


# class YOLOv8_ModList(nn.Module):
class YOLOv8_AlignedWithPT(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()

        self.model = nn.ModuleList(
            [
                # Backbone
                Conv(3, 32, 3, 2),  # 0
                Conv(32, 64, 3, 2),  # 1
                C2f(64, 64, n=1, shortcut=True),  # 2
                Conv(64, 128, 3, 2),  # 3
                C2f(128, 128, n=2, shortcut=True),  # 4
                Conv(128, 256, 3, 2),  # 5
                C2f(256, 256, n=2, shortcut=True),  # 6
                Conv(256, 512, 3, 2),  # 7
                C2f(512, 512, n=1, shortcut=True),  # 8
                SPPF(512, 512),  # 9
                # Neck - FPN
                nn.Upsample(scale_factor=2, mode="nearest"),  # 10
                Concat(),  # 11
                C2f(768, 256, n=1),  # 12
                nn.Upsample(scale_factor=2, mode="nearest"),  # 13
                Concat(),  # 14
                C2f(384, 128, n=1),  # 15
                # PAN
                Conv(128, 128, 3, 2),  # 16
                Concat(),  # 17
                C2f(384, 256, n=1),  # 18
                Conv(256, 256, 3, 2),  # 19
                Concat(),  # 20
                C2f(768, 512, n=1),  # 21
                # Official Detect head from YOLOv8
                Detect(nc=num_classes, ch=[128, 256, 512]),  # 22
            ]
        )

        for m in self.model.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.bn.eps = 0.001
                m.bn.momentum = 0.03

    def forward(self, x):
        m = self.model
        # Backbone
        x = m[0](x)
        x = m[1](x)
        x = m[2](x)

        x = m[3](x)
        x = m[4](x)
        p3 = x  # P3 output

        x = m[5](x)
        x = m[6](x)
        p4 = x  # P4 output

        x = m[7](x)
        x = m[8](x)
        x = m[9](x)
        p5 = x  # P5 output

        # FPN
        p5_up = m[10](p5)
        p4_td = m[11]([p5_up, p4])
        p4_td = m[12](p4_td)  # x_p4

        p4_up = m[13](p4_td)
        p3_td = m[14]([p4_up, p3])
        p3_td = m[15](p3_td)  # x_p3

        # PAN
        p3_down = m[16](p3_td)
        p4_out = m[17]([p3_down, p4_td])
        p4_out = m[18](p4_out)

        p4_down = m[19](p4_out)
        p5_out = m[20]([p4_down, p5])
        p5_out = m[21](p5_out)

        # Detect
        output = m[22]([p3_td, p4_out, p5_out], pre_inference=True)

        return output, [p3_td, p4_out, p5_out]

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        return self.model[-1].postprocess(preds, img, orig_imgs, **kwargs)

    def load_ckpt(self, ckpt_path, strict=False):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)  # handles YOLOv8 .pt format

        if isinstance(state, nn.Module):
            state = state.state_dict()
        elif "state_dict" in state:
            state = state["state_dict"]

        missing, unexpected = self.load_state_dict(state, strict=strict)
        print(f"✅ Loaded checkpoint: {ckpt_path}")
        if missing:
            print("⚠️  Missing keys:", missing)
        if unexpected:
            print("⚠️  Unexpected keys:", unexpected)
