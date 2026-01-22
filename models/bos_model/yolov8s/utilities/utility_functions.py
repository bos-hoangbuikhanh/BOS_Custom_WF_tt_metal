# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os

import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image

import ttnn

curr_file_path = os.path.dirname(os.path.realpath(__file__))
yolo_home_path = os.path.dirname(curr_file_path)
weights_dict = torch.load(os.path.join(yolo_home_path, "yolov8s_weights.pth"), map_location="cpu")


### Math operations ###
def _nearest_32(x):
    return math.ceil(x / 32) * 32


### Tensor conversion ###
def tt_to_torch_tensor(tt_tensor):
    tt_output = tt_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    return tt_output.to_torch()


def torch_to_tt_tensor_rm(py_tensor, device, shape=None, put_on_device=True):
    if shape is None:
        shape = list(py_tensor.size())
        while len(shape) < 4:
            shape.insert(0, 1)

    tt_tensor = ttnn.Tensor(py_tensor.reshape(shape), ttnn.bfloat16)
    if put_on_device:
        tt_tensor = tt_tensor.to(device)
    return tt_tensor


def torch_to_tt_tensor(py_tensor, device):
    shape = list(py_tensor.size())
    while len(shape) < 4:
        shape.insert(0, 1)

    tt_tensor = (
        ttnn.Tensor(py_tensor.reshape(shape), ttnn.bfloat16)
        .to(
            ttnn.TILE_LAYOUT
        )  # change memory layout of TT Tensor to TILE (as operation that will use it expects TILE layout)
        .to(device)  # move TT Tensor from host to TT accelerator device (device is of type ttnn.device.Device)
    )

    return tt_tensor


### Measuring accuracy and other metrics ###
def comp_pcc(golden, calculated, pcc=0.99):
    golden = torch.Tensor(golden)
    calculated = torch.Tensor(calculated)

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, 0.0

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.error("One tensor is all zero")
        return False, 0.0

    # For now, mask all infs and nans so that we check the rest... TODO
    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, 1.0

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, 1.0

    return cal_pcc >= pcc, cal_pcc


def dprint(printer1="", printer2="", printer3="", printer4="", printer5="", printer6=""):
    from run_yolov8s import debug_printer

    if debug_printer == True:
        print(printer1, printer2, printer3, printer4, printer5, printer6)


def divide(image_shape, quotient):
    return (image_shape[0] // quotient, image_shape[1] // quotient)


def prepare_conv_weights(base_address, conv_bias=False, batchnorm_bias=True):
    # Prepare weights
    # conv_weight = weights_dict[base_address+"conv.weight"]
    # conv_bias = weights_dict[base_address+"conv.bias"] if conv_bias else torch.zeros(conv_weight.shape[0])
    pad_value = 0
    if not conv_bias:
        conv_weight = weights_dict[base_address + "conv.weight"]
        conv_bias = torch.zeros(conv_weight.shape[0])
    else:
        conv_weight = weights_dict[base_address + "weight"]
        conv_bias = weights_dict[base_address + "bias"]
        return conv_weight, conv_bias

    # Get BN parameters
    gamma = weights_dict[base_address + "bn.weight"]
    if batchnorm_bias:
        beta = weights_dict[base_address + "bn.bias"]
    mean = weights_dict[base_address + "bn.running_mean"]
    var = weights_dict[base_address + "bn.running_var"]
    # eps = weights_dict[base_address+"bn.eps"]
    eps = 0.001

    # Compute std
    std = torch.sqrt(var + eps)

    # Fold weights
    new_weight = conv_weight * (gamma / std).reshape(-1, 1, 1, 1)

    # Fold bias
    if batchnorm_bias:
        new_bias = (conv_bias - mean) * (gamma / std) + beta
        return new_weight, new_bias
    else:
        return new_weight


def torch_pcc(x, y):
    x, y = x.flatten(), y.flatten()
    x_mean = x.mean()
    y_mean = y.mean()
    x_diff = x - x_mean
    y_diff = y - y_mean
    numerator = torch.sum(x_diff * y_diff)
    denominator = torch.sqrt(torch.sum(x_diff**2)) * torch.sqrt(torch.sum(y_diff**2))
    pcc = numerator / (denominator + 1e-8)
    return pcc.item()


def load_resize_and_pad_channels(image_path, image_size=[256, 256], golden=False):
    img = Image.open(image_path).convert("RGB")
    original_w, original_h = img.size
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
        ]
    )
    tensor = transform(img)

    c, h, w = tensor.shape
    if golden == False:
        if c % 32 != 0:
            pad_c = _nearest_32(c) - c
            pad_tensor = torch.zeros((pad_c, h, w), dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad_tensor], dim=0)

        tensor = tensor.unsqueeze(0).to(dtype=torch.bfloat16).permute(0, 2, 3, 1)  # NCHW -> NHWC
    else:
        tensor = tensor.unsqueeze(0)

    return tensor, (original_w, original_h), image_size


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

        tensor = tensor.unsqueeze(0).to(dtype=torch.bfloat16).permute(0, 2, 3, 1)  # NCHW -> NHWC
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


def pad_channels_(tensor):
    c, h, w = tensor.shape
    if c % 32 != 0:
        pad_c = _nearest_32(c) - c
        pad_tensor = torch.zeros((pad_c, h, w), dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad_tensor], dim=0)

    tensor = tensor.unsqueeze(0).to(dtype=torch.bfloat16)
    return tensor


def setup_l1_sharded_input(device, torch_input_tensor=None, min_channels=32, num_cores=20):
    if num_cores == 20:
        core_grid = ttnn.CoreGrid(y=4, x=5)
    elif num_cores == 16:
        core_grid = ttnn.CoreGrid(y=4, x=4)
    else:
        core_grid = ttnn.CoreGrid(y=8, x=8)
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
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
    )
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return tt_inputs_host, input_mem_config
