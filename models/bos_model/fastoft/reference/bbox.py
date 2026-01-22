# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib import transforms
from . import utils
import torch
import cv2
import numpy as np

def draw_bbox2d(objects, color="k", ax=None):
    limits = ax.axis()

    for obj in objects:
        x, _, z = obj.position
        l, _, w = obj.dimensions

        # Setup transform
        t = transforms.Affine2D().rotate(obj.angle + math.pi / 2)
        t = t.translate(x, z) + ax.transData

        # Draw 2D object bounding box
        rect = Rectangle((-w / 2, -l / 2), w, l, edgecolor=color, transform=t, fill=False)
        ax.add_patch(rect)

        # Draw dot indicating object center
        center = Circle((x, z), 0.25, facecolor="k")
        ax.add_patch(center)

    ax.axis(limits)
    return ax


def draw_bbox3d(obj, calib, ax, color="b"):
    # Get corners of 3D bounding box
    corners = utils.bbox_corners(obj)

    # Project into image coordinates
    img_corners = utils.perspective(calib.cpu(), corners, dtype=torch.float32).numpy()

    # Draw polygons
    # Front face
    ax.add_patch(Polygon(img_corners[[1, 3, 7, 5]], ec=color, fill=False))
    # Back face
    ax.add_patch(Polygon(img_corners[[0, 2, 6, 4]], ec=color, fill=False))
    ax.add_line(Line2D(*img_corners[[0, 1]].T, c=color))  # Lower left
    ax.add_line(Line2D(*img_corners[[2, 3]].T, c=color))  # Lower right
    ax.add_line(Line2D(*img_corners[[4, 5]].T, c=color))  # Upper left
    ax.add_line(Line2D(*img_corners[[6, 7]].T, c=color))  # Upper right


def visualize_objects(image, calib, objects, cmap="tab20", ax=None):
    # Create a figure if it doesn't already exist
    if ax is None:
        fig, ax = plt.subplots()
    ax.clear()

    # Visualize image
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    extents = ax.axis()

    # Visualize objects
    # Check if cmap is a color string (like "#00D9FF") or a colormap name
    if isinstance(cmap, str) and (cmap.startswith("#") or cmap in ["r", "g", "b", "c", "m", "y", "k", "w"]):
        # Use single color for all objects
        color = cmap
        for i, obj in enumerate(objects):
            draw_bbox3d(obj, calib, ax, color)
    else:
        # Use colormap
        cmap = cm.get_cmap(cmap, len(objects))
        for i, obj in enumerate(objects):
            draw_bbox3d(obj, calib, ax, cmap(i))

    # Format axis
    ax.axis(extents)
    ax.axis(False)
    ax.grid(False)
    return ax


def _mpl_to_bgr255(rgba):
    r, g, b, _ = rgba
    return (int(b * 255), int(g * 255), int(r * 255))


def _apply_prep_config(points, prep_config):
    if prep_config is None:
        return points
    scale_x = prep_config.get("scale_x", 1.0)
    scale_y = prep_config.get("scale_y", 1.0)
    x_offset = prep_config.get("x_offset", 0.0)
    y_offset = prep_config.get("y_offset", 0.0)

    # Validate scale values to avoid division by zero when reversing preprocessing
    if scale_x == 0 or scale_y == 0:
        raise ValueError(f"prep_config scale_x and scale_y must be non-zero, got scale_x={scale_x}, scale_y={scale_y}")
    points = points.copy()
    points[:, 0] = (points[:, 0] - x_offset) / scale_x
    points[:, 1] = (points[:, 1] - y_offset) / scale_y
    return points


def draw_bbox3d_cv(img_bgr, obj, calib, color=(0, 255, 0), thickness=2, prep_config=None):
    # Get 3D corners and project to image plane
    corners = utils.bbox_corners(obj)
    img_corners = utils.perspective(calib.cpu(), corners, dtype=torch.float32).numpy()
    img_corners = _apply_prep_config(img_corners, prep_config)

    pts = np.round(img_corners).astype(int)
    front = pts[[1, 3, 7, 5]]
    back = pts[[0, 2, 6, 4]]

    # Draw faces
    cv2.polylines(
        img_bgr, [front.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA
    )
    cv2.polylines(
        img_bgr, [back.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA
    )

    # Connect corresponding corners
    for a, b in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        cv2.line(img_bgr, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)

    return img_bgr


def visualize_objects_cv(image, calib, objects, cmap="tab20", thickness=2, prep_config=None):
    # Convert input image to uint8 RGB numpy
    if hasattr(image, "detach"):  # torch tensor
        img_rgb = image.detach().cpu()
        if img_rgb.dim() == 3 and img_rgb.shape[0] in (3, 4):  # C,H,W
            img_rgb = img_rgb[:3].permute(1, 2, 0).numpy()
        else:
            raise ValueError("Expected image tensor with shape (C,H,W).")
        if img_rgb.dtype != np.uint8:
            img_rgb = np.clip(img_rgb * (255.0 if img_rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    else:
        img_rgb = np.asarray(image)
        if img_rgb.dtype != np.uint8:
            img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()

    # Colors from a matplotlib colormap (distinct per object)
    if isinstance(cmap, tuple):
        color = cmap
        for obj in objects:
            draw_bbox3d_cv(img_bgr, obj, calib, color=color, thickness=thickness, prep_config=prep_config)
    else:
        cmap = cm.get_cmap(cmap, max(len(objects), 1))
        for i, obj in enumerate(objects):
            color = _mpl_to_bgr255(cmap(i))
            draw_bbox3d_cv(img_bgr, obj, calib, color=color, thickness=thickness, prep_config=prep_config)

    return img_bgr