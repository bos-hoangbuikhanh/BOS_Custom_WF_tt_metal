import sys
from collections import Counter
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def find_instance_center(center_heatmap, threshold=0.1, nms_kernel=3, top_k=None):
    """
    Find the center points from the center heatmap.
    Args:
        center_heatmap: A Tensor of shape [1, H, W] of raw center heatmap output.
        threshold: A float, threshold applied to center heatmap score.
        nms_kernel: An integer, NMS max pooling kernel size.
        top_k: An integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The
            order of second dim is (y, x).
    """
    # Thresholding, setting values below threshold to -1.
    center_heatmap = F.threshold(center_heatmap, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    center_heatmap_max_pooled = F.max_pool2d(center_heatmap, kernel_size=nms_kernel, stride=1, padding=nms_padding)
    center_heatmap[center_heatmap != center_heatmap_max_pooled] = -1

    # Squeeze first two dimensions.
    center_heatmap = center_heatmap.squeeze()
    assert len(center_heatmap.size()) == 2, "Something is wrong with center heatmap dimension."

    # Find non-zero elements.
    if top_k is None:
        return torch.nonzero(center_heatmap > 0)
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(center_heatmap), top_k)
        return torch.nonzero(center_heatmap > top_k_scores[-1].clamp_(min=0))


def group_pixels(center_points, offsets):
    height, width = offsets.size()[1:]
    y_coord, x_coord = torch.meshgrid(
        torch.arange(height, dtype=offsets.dtype, device=offsets.device),
        torch.arange(width, dtype=offsets.dtype, device=offsets.device),
    )
    coord = torch.cat((y_coord.unsqueeze(0), x_coord.unsqueeze(0)), dim=0)

    center_loc = coord + offsets
    center_loc = center_loc.flatten(1).T.unsqueeze_(0)  # [1, H*W, 2]
    center_points = center_points.unsqueeze(1)  # [K, 1, 2]

    # Distance: [K, H*W].
    distance = torch.norm(center_points - center_loc, dim=-1)
    instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1
    return instance_id


def get_instance_segmentation(
    sem_seg, center_heatmap, offsets, thing_seg, thing_ids, threshold=0.1, nms_kernel=3, top_k=None
):
    center_points = find_instance_center(center_heatmap, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k)
    if center_points.size(0) == 0:
        return torch.zeros_like(sem_seg), center_points.unsqueeze(0)
    ins_seg = group_pixels(center_points, offsets)
    return thing_seg * ins_seg, center_points.unsqueeze(0)


def merge_semantic_and_instance(sem_seg, ins_seg, semantic_thing_seg, label_divisor, thing_ids, stuff_area, void_label):
    # In case thing mask does not align with semantic prediction.
    pan_seg = torch.zeros_like(sem_seg) + void_label
    is_thing = (ins_seg > 0) & (semantic_thing_seg > 0)

    # Keep track of instance id for each class.
    class_id_tracker = Counter()

    # Paste thing by majority voting.
    instance_ids = torch.unique(ins_seg)
    for ins_id in instance_ids:
        if ins_id == 0:
            continue
        # Make sure only do majority voting within `semantic_thing_seg`.
        thing_mask = (ins_seg == ins_id) & is_thing
        if torch.nonzero(thing_mask).size(0) == 0:
            continue
        class_id, _ = torch.mode(sem_seg[thing_mask].view(-1))
        class_id_tracker[class_id.item()] += 1
        new_ins_id = class_id_tracker[class_id.item()]
        pan_seg[thing_mask] = class_id * label_divisor + new_ins_id

    # Paste stuff to unoccupied area.
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id.item() in thing_ids:
            # thing class
            continue
        # Calculate stuff area.
        stuff_mask = (sem_seg == class_id) & (ins_seg == 0)
        if stuff_mask.sum().item() >= stuff_area:
            pan_seg[stuff_mask] = class_id * label_divisor

    return pan_seg


def get_panoptic_segmentation(
    sem_seg,
    center_heatmap,
    offsets,
    thing_ids,
    label_divisor,
    stuff_area,
    void_label,
    threshold=0.1,
    nms_kernel=7,
    top_k=200,
    foreground_mask=None,
):
    if sem_seg.dim() != 3 and sem_seg.size(0) != 1:
        raise ValueError("Semantic prediction with un-supported shape: {}.".format(sem_seg.size()))
    if center_heatmap.dim() != 3:
        raise ValueError("Center prediction with un-supported dimension: {}.".format(center_heatmap.dim()))
    if offsets.dim() != 3:
        raise ValueError("Offset prediction with un-supported dimension: {}.".format(offsets.dim()))
    if foreground_mask is not None:
        if foreground_mask.dim() != 3 and foreground_mask.size(0) != 1:
            raise ValueError("Foreground prediction with un-supported shape: {}.".format(sem_seg.size()))
        thing_seg = foreground_mask
    else:
        # inference from semantic segmentation
        thing_seg = torch.zeros_like(sem_seg)
        for thing_class in list(thing_ids):
            thing_seg[sem_seg == thing_class] = 1

    instance, center = get_instance_segmentation(
        sem_seg,
        center_heatmap,
        offsets,
        thing_seg,
        thing_ids,
        threshold=threshold,
        nms_kernel=nms_kernel,
        top_k=top_k,
    )
    panoptic = merge_semantic_and_instance(
        sem_seg, instance, thing_seg, label_divisor, thing_ids, stuff_area, void_label
    )

    return panoptic, center


def sem_seg_postprocess(result, img_size, output_height, output_width):
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(result, size=(output_height, output_width), mode="bilinear", align_corners=False)[0]
    return result


class ResizeTransform:
    def __init__(self, h, w, new_h, new_w, interp=None):
        if interp is None:
            interp = Image.BILINEAR

        self.h = h
        self.w = w
        self.new_h = new_h
        self.new_w = new_w
        self.interp = interp

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode="L")
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: "nearest",
                Image.BILINEAR: "bilinear",
                Image.BICUBIC: "bicubic",
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == "nearest" else False
            img = F.interpolate(img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners)
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)


class ResizeShortestEdge:
    def __init__(self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR):
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"

        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.sample_style = sample_style
        self.interp = interp

        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style." f" Got {short_edge_length}!"
            )
        # self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)

        # if size == 0:
        #     return NoOpTransform()

        newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, self.max_size)
        return ResizeTransform(h, w, newh, neww, self.interp)

    @staticmethod
    def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
