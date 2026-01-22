from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from models.bos_model.pdl.reference.norm_utils import BitMasks, ImageList, Instances, ShapeSpec, configurable, get_norm
from models.bos_model.pdl.reference.post_processing import get_panoptic_segmentation, sem_seg_postprocess

# from detectron2.data import MetadataCatalog


class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DeepLabStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, norm="BN"):
        super().__init__()
        self.stride = 4
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels // 2),
        )
        self.conv2 = Conv2d(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels // 2),
        )
        self.conv3 = Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        # weight_init.c2_msra_fill(self.conv1)
        # weight_init.c2_msra_fill(self.conv2)
        # weight_init.c2_msra_fill(self.conv3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = F.relu_(x)
        x = self.conv3(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class CNNBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride


class BottleneckBlock(CNNBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class Backbone(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @property
    def size_divisibility(self) -> int:
        return 0

    @property
    def padding_constraints(self) -> Dict[str, int]:
        return {}

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }


class ResNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            num_stages = max([{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features])
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(current_stride * np.prod([k.stride for k in blocks]))
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        self.freeze(freeze_at)

    def forward(self, x):
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs):
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the " f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs))
            in_channels = out_channels
        return blocks


def build_resnet_deeplab_backbone(cfg, input_shape):
    norm = cfg.MODEL.RESNETS.NORM

    if cfg.MODEL.RESNETS.STEM_TYPE == "deeplab":
        stem = DeepLabStem(
            in_channels=input_shape.channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            norm=norm,
        )
    else:
        raise ValueError("Unknown stem type: {}".format(cfg.MODEL.RESNETS.STEM_TYPE))

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res4_dilation       = cfg.MODEL.RESNETS.RES4_DILATION
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    res5_multi_grid     = cfg.MODEL.RESNETS.RES5_MULTI_GRID

    # fmt: on
    assert res4_dilation in {1, 2}, "res4_dilation cannot be {}.".format(res4_dilation)
    assert res5_dilation in {1, 2, 4}, "res5_dilation cannot be {}.".format(res5_dilation)
    if res4_dilation == 2:
        # Always dilate res5 if res4 is dilated.
        assert res5_dilation == 4

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        if stage_idx == 4:
            dilation = res4_dilation
        elif stage_idx == 5:
            dilation = res5_dilation
        else:
            dilation = 1
        first_stride = 1 if idx == 0 or dilation > 1 else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["stride_in_1x1"] = stride_in_1x1
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
        stage_kargs["block_class"] = BottleneckBlock

        if stage_idx == 5:
            stage_kargs.pop("dilation")
            stage_kargs["dilation_per_block"] = [dilation * mg for mg in res5_multi_grid]

        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features)


def build_backbone(cfg, input_shape=None):
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = build_resnet_deeplab_backbone(cfg, input_shape)
    return backbone


def build_sem_seg_head(cfg, input_shape):
    """
    Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return PanopticDeepLabSemSegHead(cfg, input_shape)


from types import SimpleNamespace


def get_cityscapes_metadata():
    return SimpleNamespace(
        name="cityscapes_fine_panoptic_train",
        panoptic_root="datasets/cityscapes/gtFine/cityscapes_panoptic_train",
        image_root="datasets/cityscapes/leftImg8bit/train",
        panoptic_json="datasets/cityscapes/gtFine/cityscapes_panoptic_train.json",
        gt_dir="datasets/cityscapes/gtFine/train",
        evaluator_type="cityscapes_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        thing_classes=[
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ],
        thing_colors=[
            (128, 64, 128),
            (244, 35, 232),
            (70, 70, 70),
            (102, 102, 156),
            (190, 153, 153),
            (153, 153, 153),
            (250, 170, 30),
            (220, 220, 0),
            (107, 142, 35),
            (152, 251, 152),
            (70, 130, 180),
            (220, 20, 60),
            (255, 0, 0),
            (0, 0, 142),
            (0, 0, 70),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 230),
            (119, 11, 32),
        ],
        stuff_classes=[
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ],
        stuff_colors=[
            (128, 64, 128),
            (244, 35, 232),
            (70, 70, 70),
            (102, 102, 156),
            (190, 153, 153),
            (153, 153, 153),
            (250, 170, 30),
            (220, 220, 0),
            (107, 142, 35),
            (152, 251, 152),
            (70, 130, 180),
            (220, 20, 60),
            (255, 0, 0),
            (0, 0, 142),
            (0, 0, 70),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 230),
            (119, 11, 32),
        ],
        thing_dataset_id_to_contiguous_id={24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18},
        stuff_dataset_id_to_contiguous_id={7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10},
    )


class PanopticDeepLab(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.ins_embed_head = build_ins_embed_branch(cfg, self.backbone.output_shape())
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        # self.meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.meta = get_cityscapes_metadata()
        self.stuff_area = cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA
        self.threshold = cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD
        self.nms_kernel = cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL
        self.top_k = cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE
        self.predict_instances = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES
        self.use_depthwise_separable_conv = cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        assert (
            cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
            == cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        )
        self.size_divisibility = cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY
        self.benchmark_network_speed = cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # To avoid error in ASPP layer when input has different size.
        size_divisibility = self.size_divisibility if self.size_divisibility > 0 else self.backbone.size_divisibility
        images = ImageList.from_tensors(images, size_divisibility)
        features = self.backbone(images.tensor)

        losses = {}
        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(targets, size_divisibility, self.sem_seg_head.ignore_value).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                weights = ImageList.from_tensors(weights, size_divisibility).tensor
            else:
                weights = None
        else:
            targets = None
            weights = None
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, targets, weights)

        losses.update(sem_seg_losses)

        if "center" in batched_inputs[0] and "offset" in batched_inputs[0]:
            center_targets = [x["center"].to(self.device) for x in batched_inputs]
            center_targets = ImageList.from_tensors(center_targets, size_divisibility).tensor.unsqueeze(1)
            center_weights = [x["center_weights"].to(self.device) for x in batched_inputs]
            center_weights = ImageList.from_tensors(center_weights, size_divisibility).tensor

            offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
            offset_targets = ImageList.from_tensors(offset_targets, size_divisibility).tensor
            offset_weights = [x["offset_weights"].to(self.device) for x in batched_inputs]
            offset_weights = ImageList.from_tensors(offset_weights, size_divisibility).tensor
        else:
            center_targets = None
            center_weights = None

            offset_targets = None
            offset_weights = None

        center_results, offset_results, center_losses, offset_losses = self.ins_embed_head(
            features, center_targets, center_weights, offset_targets, offset_weights
        )

        losses.update(center_losses)
        losses.update(offset_losses)

        if self.training:
            return losses

        if self.benchmark_network_speed:
            return []

        processed_results = []
        for sem_seg_result, center_result, offset_result, input_per_image, image_size in zip(
            sem_seg_results, center_results, offset_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            c = sem_seg_postprocess(center_result, image_size, height, width)
            o = sem_seg_postprocess(offset_result, image_size, height, width)
            # Post-processing to get panoptic segmentation.
            panoptic_image, _ = get_panoptic_segmentation(
                r.argmax(dim=0, keepdim=True),
                c,
                o,
                thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
                label_divisor=self.meta.label_divisor,
                stuff_area=self.stuff_area,
                void_label=-1,
                threshold=self.threshold,
                nms_kernel=self.nms_kernel,
                top_k=self.top_k,
            )
            # For semantic segmentation evaluation.
            processed_results.append({"sem_seg": r})
            panoptic_image = panoptic_image.squeeze(0)
            semantic_prob = F.softmax(r, dim=0)
            # For panoptic segmentation evaluation.
            processed_results[-1]["panoptic_seg"] = (panoptic_image, None)
            # For instance segmentation evaluation.
            if self.predict_instances:
                instances = []
                panoptic_image_cpu = panoptic_image.cpu().numpy()
                for panoptic_label in np.unique(panoptic_image_cpu):
                    if panoptic_label == -1:
                        continue
                    pred_class = panoptic_label // self.meta.label_divisor
                    isthing = pred_class in list(self.meta.thing_dataset_id_to_contiguous_id.values())
                    # Get instance segmentation results.
                    if isthing:
                        instance = Instances((height, width))
                        # Evaluation code takes continuous id starting from 0
                        instance.pred_classes = torch.tensor([pred_class], device=panoptic_image.device)
                        mask = panoptic_image == panoptic_label
                        instance.pred_masks = mask.unsqueeze(0)
                        # Average semantic probability
                        sem_scores = semantic_prob[pred_class, ...]
                        sem_scores = torch.mean(sem_scores[mask])
                        # Center point probability
                        mask_indices = torch.nonzero(mask).float()
                        center_y, center_x = (
                            torch.mean(mask_indices[:, 0]),
                            torch.mean(mask_indices[:, 1]),
                        )
                        center_scores = c[0, int(center_y.item()), int(center_x.item())]
                        # Confidence score is semantic prob * center prob.
                        instance.scores = torch.tensor([sem_scores * center_scores], device=panoptic_image.device)
                        # Get bounding boxes
                        instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
                        instances.append(instance)
                if len(instances) > 0:
                    processed_results[-1]["instances"] = Instances.cat(instances)

        return processed_results


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dilations,
        *,
        norm,
        activation,
        pool_kernel_size=None,
        dropout: float = 0.0,
        use_depthwise_separable_conv=False,
    ):
        super(ASPP, self).__init__()
        assert len(dilations) == 3, "ASPP expects 3 dilations, got {}".format(len(dilations))
        self.pool_kernel_size = pool_kernel_size
        self.dropout = dropout
        use_bias = norm == ""
        self.convs = nn.ModuleList()
        # conv 1x1
        self.convs.append(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=use_bias,
                norm=get_norm(norm, out_channels),
                activation=deepcopy(activation),
            )
        )
        # weight_init.c2_xavier_fill(self.convs[-1])
        # atrous convs
        for dilation in dilations:
            self.convs.append(
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=use_bias,
                    norm=get_norm(norm, out_channels),
                    activation=deepcopy(activation),
                )
            )

        if pool_kernel_size is None:
            image_pooling = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv2d(in_channels, out_channels, 1, bias=True, activation=deepcopy(activation)),
            )
        else:
            image_pooling = nn.Sequential(
                nn.AvgPool2d(kernel_size=pool_kernel_size, stride=1),
                Conv2d(in_channels, out_channels, 1, bias=True, activation=deepcopy(activation)),
            )
        self.convs.append(image_pooling)

        self.project = Conv2d(
            5 * out_channels,
            out_channels,
            kernel_size=1,
            bias=use_bias,
            norm=get_norm(norm, out_channels),
            activation=deepcopy(activation),
        )
        # weight_init.c2_xavier_fill(self.project)

    def forward(self, x):
        size = x.shape[-2:]
        if self.pool_kernel_size is not None:
            if size[0] % self.pool_kernel_size[0] or size[1] % self.pool_kernel_size[1]:
                raise ValueError(
                    "`pool_kernel_size` must be divisible by the shape of inputs. "
                    "Input size: {} `pool_kernel_size`: {}".format(size, self.pool_kernel_size)
                )
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[-1] = F.interpolate(res[-1], size=size, mode="bilinear", align_corners=False)
        res = torch.cat(res, dim=1)
        res = self.project(res)
        # res = F.dropout(res, self.dropout, training=self.training) if self.dropout > 0 else res
        return res


class DeepLabV3PlusHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
        common_stride: int,
        norm: Union[str, Callable],
        train_size: Optional[Tuple],
        loss_weight: float = 1.0,
        loss_type: str = "cross_entropy",
        ignore_value: int = -1,
        num_classes: Optional[int] = None,
        use_depthwise_separable_conv: bool = False,
    ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)

        # fmt: off
        self.in_features      = [k for k, v in input_shape]  # starting from "res2" to "res5"
        in_channels           = [x[1].channels for x in input_shape]
        in_strides            = [x[1].stride for x in input_shape]
        aspp_channels         = decoder_channels[-1]
        self.ignore_value     = ignore_value
        self.common_stride    = common_stride  # output stride
        self.loss_weight      = loss_weight
        self.loss_type        = loss_type
        self.decoder_only     = num_classes is None
        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        # fmt: on

        assert len(project_channels) == len(self.in_features) - 1, "Expected {} project_channels, got {}".format(
            len(self.in_features) - 1, len(project_channels)
        )
        assert len(decoder_channels) == len(self.in_features), "Expected {} decoder_channels, got {}".format(
            len(self.in_features), len(decoder_channels)
        )
        self.decoder = nn.ModuleDict()

        use_bias = norm == ""
        for idx, in_channel in enumerate(in_channels):
            decoder_stage = nn.ModuleDict()

            if idx == len(self.in_features) - 1:
                # ASPP module
                if train_size is not None:
                    train_h, train_w = train_size
                    encoder_stride = in_strides[-1]
                    if train_h % encoder_stride or train_w % encoder_stride:
                        raise ValueError("Crop size need to be divisible by encoder stride.")
                    pool_h = train_h // encoder_stride
                    pool_w = train_w // encoder_stride
                    pool_kernel_size = (pool_h, pool_w)
                else:
                    pool_kernel_size = None
                project_conv = ASPP(
                    in_channel,
                    aspp_channels,
                    aspp_dilations,
                    norm=norm,
                    activation=F.relu,
                    pool_kernel_size=pool_kernel_size,
                    dropout=aspp_dropout,
                    use_depthwise_separable_conv=use_depthwise_separable_conv,
                )
                fuse_conv = None
            else:
                project_conv = Conv2d(
                    in_channel,
                    project_channels[idx],
                    kernel_size=1,
                    bias=use_bias,
                    norm=get_norm(norm, project_channels[idx]),
                    activation=F.relu,
                )
                # weight_init.c2_xavier_fill(project_conv)

                fuse_conv = nn.Sequential(
                    Conv2d(
                        project_channels[idx] + decoder_channels[idx + 1],
                        decoder_channels[idx],
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, decoder_channels[idx]),
                        activation=F.relu,
                    ),
                    Conv2d(
                        decoder_channels[idx],
                        decoder_channels[idx],
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, decoder_channels[idx]),
                        activation=F.relu,
                    ),
                )
                # weight_init.c2_xavier_fill(fuse_conv[0])
                # weight_init.c2_xavier_fill(fuse_conv[1])

            decoder_stage["project_conv"] = project_conv
            decoder_stage["fuse_conv"] = fuse_conv

            self.decoder[self.in_features[idx]] = decoder_stage

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM] * (len(cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES) - 1) + [
            cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS
        ]
        ret = dict(
            input_shape={k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES},
            project_channels=cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.SEM_SEG_HEAD.NORM,
            train_size=train_size,
            loss_weight=cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            loss_type=cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE,
            ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def layers(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for f in self.in_features[::-1]:
            x = features[f]
            proj_x = self.decoder[f]["project_conv"](x)
            if self.decoder[f]["fuse_conv"] is None:
                # This is aspp module
                y = proj_x
            else:
                # Upsample y
                y = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
                y = torch.cat([proj_x, y], dim=1)
                y = self.decoder[f]["fuse_conv"](y)
        if not self.decoder_only:
            y = self.predictor(y)
        return y

    def losses(self, predictions, targets):
        predictions = F.interpolate(predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


class PanopticDeepLabSemSegHead(DeepLabV3PlusHead):
    """
    A semantic segmentation head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        loss_weight: float,
        loss_type: str,
        loss_top_k: float,
        ignore_value: int,
        num_classes: int,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )
        assert self.decoder_only

        self.loss_weight = loss_weight
        use_bias = norm == ""

        self.head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        # weight_init.c2_xavier_fill(self.head[0])
        # weight_init.c2_xavier_fill(self.head[1])

        self.predictor = Conv2d(head_channels, num_classes, kernel_size=1)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        ret["loss_top_k"] = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        return ret

    def forward(self, features, targets=None, weights=None):
        y = self.layers(features)
        if self.training:
            return None, self.losses(y, targets, weights)
        else:
            y = F.interpolate(y, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
            return y, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        y = self.head(y)
        y = self.predictor(y)
        return y

    def losses(self, predictions, targets, weights=None):
        predictions = F.interpolate(predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        loss = self.loss(predictions, targets, weights)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


def build_ins_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return PanopticDeepLabInsEmbedHead(cfg, input_shape)


class PanopticDeepLabInsEmbedHead(DeepLabV3PlusHead):
    """
    A instance embedding head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        center_loss_weight: float,
        offset_loss_weight: float,
        **kwargs,
    ):
        super().__init__(input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs)
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        use_bias = norm == ""
        # center prediction
        # `head` is additional transform before predictor
        self.center_head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )

        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)

        self.offset_head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )

        # weight_init.c2_xavier_fill(self.offset_head[0])
        # weight_init.c2_xavier_fill(self.offset_head[1])

        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)

        # nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        # nn.init.constant_(self.offset_predictor.bias, 0)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1) + [
            cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS
        ]
        ret = dict(
            input_shape={k: v for k, v in input_shape.items() if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES},
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(
        self,
        features,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
    ):
        center, offset = self.layers(features)
        if self.training:
            return (
                None,
                None,
                self.center_losses(center, center_targets, center_weights),
                self.offset_losses(offset, offset_targets, offset_weights),
            )
        else:
            center = F.interpolate(center, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
            offset = (
                F.interpolate(offset, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
                * self.common_stride
            )
            return center, offset, {}, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        # center
        center = self.center_head(y)
        center = self.center_predictor(center)
        # offset
        offset = self.offset_head(y)
        offset = self.offset_predictor(offset)
        return center, offset

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
            * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses
