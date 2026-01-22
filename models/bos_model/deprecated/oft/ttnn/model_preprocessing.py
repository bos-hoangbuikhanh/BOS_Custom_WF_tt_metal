import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.bos_model.oft.reference.oftnet import OftNet


def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout)
    return weight


def preprocess_linear_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout)
    return bias


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
    if isinstance(model, torch.nn.Linear):
        parameters[f"weight"] = preprocess_linear_weight(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            parameters[f"bias"] = preprocess_linear_bias(model.bias, dtype=ttnn.float32)
    if isinstance(model, nn.GroupNorm):
        parameters["weight"] = model.weight
        if model.bias is not None:
            parameters["bias"] = model.bias

    return parameters


def create_OFT_model_parameters_resnet(model: OftNet, input_tensor: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    parameters["model_args"] = model

    return parameters


def create_OFT_model_parameters_oft(
    model: OftNet,
    input_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device,
):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    input1, input2, input3 = input_tensors
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(input1, input2, input3),
        device=None,
    )

    parameters["model_args"] = model

    return parameters


def create_OFT_model_parameters(
    model: OftNet,
    input_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device,
):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )
    parameters.oft8.conv3d.weight = ttnn.to_device(parameters.oft8.conv3d.weight, device=device)
    parameters.oft8.conv3d.bias = ttnn.to_device(parameters.oft8.conv3d.bias, device=device)
    parameters.oft16.conv3d.weight = ttnn.to_device(parameters.oft16.conv3d.weight, device=device)
    parameters.oft16.conv3d.bias = ttnn.to_device(parameters.oft16.conv3d.bias, device=device)
    parameters.oft32.conv3d.weight = ttnn.to_device(parameters.oft32.conv3d.weight, device=device)
    parameters.oft32.conv3d.bias = ttnn.to_device(parameters.oft32.conv3d.bias, device=device)

    input1, input2, input3 = input_tensors
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(input1, input2, input3),
        device=None,
    )

    parameters["model_args"] = model

    return parameters


def perspective(matrix, vector):
    vector = vector.unsqueeze(-1)
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = homogenous.squeeze(-1)
    homogenous = homogenous[..., :-1] / homogenous[..., [-1]]
    return homogenous


def preprocess_calib_grid_scale(calib, grid, grid_height, cell_size, scale, lat_shape, device=None):
    EPSILON = 1e-6

    y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])

    corners = grid.unsqueeze(1) + y_corners.view(-1, 1, 1, 3)
    img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)

    img_height, img_width = lat_shape  # Differs for resolution

    img_size = corners.new([img_width, img_height]) / scale
    norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)

    bbox_corners = torch.cat(
        [
            torch.min(norm_corners[:, :-1, :-1, :-1], norm_corners[:, :-1, 1:, :-1]),
            torch.max(norm_corners[:, 1:, 1:, 1:], norm_corners[:, 1:, :-1, 1:]),
        ],
        dim=-1,
    )

    batch, _, depth, width, _ = bbox_corners.size()
    bbox_corners = bbox_corners.flatten(2, 3)

    area = (
        (bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) * img_height * img_width * 0.25 + EPSILON
    ).unsqueeze(1)

    visible = area > EPSILON
    visible = visible.float()
    visible = ttnn.from_torch(visible, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    area = ttnn.from_torch(area, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    pre_config = {
        "visible": visible,  # ttnn tensor
        "area": area,
        "bbox_corners_shape": [batch, _, depth, width, _],
        "bbox_corners": bbox_corners,
    }
    return pre_config


def preprocessing(calib, grid, grid_height, cell_size, res=(384, 1248), device=None):
    # For res 384*1248
    if res == (384, 1248):
        lat8_shape, lat16_shape, lat32_shape = (48, 156), (24, 78), (12, 39)
    elif res == (256, 832):
        lat8_shape, lat16_shape, lat32_shape = (32, 104), (16, 52), (8, 26)
    elif res == (180, 540):
        lat8_shape, lat16_shape, lat32_shape = (23, 68), (12, 34), (6, 17)
    else:
        raise ValueError("Resolution is not supported by the model")

    return {
        "oft8": preprocess_calib_grid_scale(calib, grid, grid_height, cell_size, 1 / 8.0, lat8_shape, device=device),
        "oft16": preprocess_calib_grid_scale(calib, grid, grid_height, cell_size, 1 / 16.0, lat16_shape, device=device),
        "oft32": preprocess_calib_grid_scale(calib, grid, grid_height, cell_size, 1 / 32.0, lat32_shape, device=device),
    }


def prepare_ttnn_input(image, device=None):
    # Demo is better if preprocess is done in TTNN
    mean = ttnn.from_torch(
        torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    std = ttnn.from_torch(
        torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    image = ttnn.from_torch(image, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    image = ttnn.subtract(image, mean, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    image = ttnn.div(image, std, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

    if image.get_layout() == ttnn.TILE_LAYOUT:
        image = ttnn.to_layout(
            image,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
    image = ttnn.permute(image, (0, 2, 3, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    n, h, w, c = image.shape
    image = ttnn.reshape(image, (1, 1, n * h * w, c))
    return image


def _nearest_32_per_core(x, core):
    return math.ceil(x / core / 32) * 32 * core


# GroupNorm Configs

gn_configs = {
    # frontend
    "frontend.bn1": {"num_out_blocks": 10, "grid_size": ttnn.CoreGrid(y=2, x=4)},
    # Layer1
    "frontend.layer1.0.bn1": {"num_out_blocks": 10, "grid_size": ttnn.CoreGrid(y=2, x=5)},
    "frontend.layer1.0.bn2": {"num_out_blocks": 10, "grid_size": ttnn.CoreGrid(y=2, x=5)},
    "frontend.layer1.1.bn1": {"num_out_blocks": 10, "grid_size": ttnn.CoreGrid(y=2, x=5)},
    "frontend.layer1.1.bn2": {"num_out_blocks": 10, "grid_size": ttnn.CoreGrid(y=2, x=5)},
    # Layer2
    "frontend.layer2.0.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer2.0.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer2.0.downsample.1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer2.1.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer2.1.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    # Layer3
    "frontend.layer3.0.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer3.0.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer3.0.downsample.1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer3.1.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer3.1.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    # Layer4
    "frontend.layer4.0.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer4.0.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer4.0.downsample.1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer4.1.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "frontend.layer4.1.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    # Lateral connections
    "bn8": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "bn16": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "bn32": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=1, x=5)},
    # Topdown path
    "topdown.0.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.0.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.1.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.1.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.2.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.2.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.3.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.3.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.4.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.4.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.5.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.5.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.6.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.6.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.7.bn1": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
    "topdown.7.bn2": {"num_out_blocks": 4, "grid_size": ttnn.CoreGrid(y=4, x=4)},
}
