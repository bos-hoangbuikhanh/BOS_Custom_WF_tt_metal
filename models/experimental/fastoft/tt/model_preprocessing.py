# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args, make_parameter_dict

from loguru import logger


def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout)
    return weight


def preprocess_linear_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout)
    return bias

def fuse_imagenet_normalization(model, mean, std):
    """
    Fuse ImageNet normalization constants into conv1 weights of a PyTorch model.
    This is the single point where normalization fusion happens.

    Args:
        model: PyTorch PanopticDeepLab model (or any model with frontend.conv1)

    Returns:
        bool: True if fusion was successful, False otherwise
    """
    if not (hasattr(model, "frontend") and hasattr(model.frontend, "conv1")):
        logger.warning("Could not find frontend.conv1 to fuse normalization")
        return False

    conv1 = model.frontend.conv1
    conv_weight = conv1.weight.data.clone()
    conv_bias = conv1.bias.data.clone() if conv1.bias is not None else None

    logger.info("Fusing image normalization into PyTorch frontend.conv1...")

    # Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
    out_channels, in_channels, kernel_h, kernel_w = conv_weight.shape
    assert in_channels == 3, f"Expected 3 input channels (RGB), got {in_channels}"

    # Store original weight for bias calculation (before scaling)
    weight_original = conv_weight.clone()

    # Scale weights: for each input channel, divide by std[channel]
    for channel_idx in range(3):
        conv_weight[:, channel_idx, :, :] = conv_weight[:, channel_idx, :, :] / std[channel_idx]

    # Adjust bias: new_bias = old_bias - sum_over_input_channels (W_original * mean / std)
    if conv_bias is None or conv_bias.numel() == 0:
        conv_bias = torch.zeros(out_channels, device=conv_weight.device, dtype=conv_weight.dtype)
    else:
        conv_bias = conv_bias.to(device=conv_weight.device, dtype=conv_weight.dtype)

    bias_adjustment = torch.zeros(out_channels, device=conv_weight.device, dtype=conv_weight.dtype)
    for out_channel in range(out_channels):
        for in_channel in range(3):
            # Sum over spatial dimensions: mean[in] / std[in] * sum(W_original[out, in, :, :])
            weight_sum = weight_original[out_channel, in_channel, :, :].sum()
            bias_adjustment[out_channel] += weight_sum * mean[in_channel] / std[in_channel]

    fused_bias = conv_bias - bias_adjustment

    # Update the layer weights (note: conv1 originally has bias=False, but we need to add bias)
    conv1.weight.data = conv_weight
    if conv1.bias is None:
        # Add bias parameter if it doesn't exist
        conv1.bias = nn.Parameter(fused_bias)
    else:
        conv1.bias.data = fused_bias

    logger.info(
        f"ImageNet normalization fusion completed: scaled weights per input channel, adjusted bias for {out_channels} output channels"
    )
    return True

def fuse_conv_bn_weights_ttnn(
    conv_weight_ttnn, conv_bias_ttnn, bn_weight_ttnn, bn_bias_ttnn, bn_running_mean_ttnn, bn_running_var_ttnn, eps=1e-5
):
    """
    Fuse convolution and batch normalization weights for TTNN tensors.

    This function implements the weight fusion described in:
    https://medium.com/@sim30217/fusing-convolution-with-batch-normalization-f9fe13b3c111

    The mathematical formulation is:
    - new_weight = conv_weight * (bn_weight / sqrt(bn_running_var + eps))
    - new_bias = (conv_bias - bn_running_mean) * (bn_weight / sqrt(bn_running_var + eps)) + bn_bias

    Args:
        conv_weight_ttnn (ttnn.Tensor): Convolution weight tensor
        conv_bias_ttnn (ttnn.Tensor or None): Convolution bias tensor or None
        bn_weight_ttnn (ttnn.Tensor): BatchNorm weight (gamma) tensor
        bn_bias_ttnn (ttnn.Tensor): BatchNorm bias (beta) tensor
        bn_running_mean_ttnn (ttnn.Tensor): BatchNorm running mean tensor
        bn_running_var_ttnn (ttnn.Tensor): BatchNorm running variance tensor
        eps (float): BatchNorm epsilon value for numerical stability

    Returns:
        tuple: (fused_weight_ttnn, fused_bias_ttnn) where both are ttnn.Tensors
    """
    # Convert TTNN tensors to PyTorch tensors for computation
    conv_weight = ttnn.to_torch(conv_weight_ttnn, dtype=torch.float32)
    conv_bias = ttnn.to_torch(conv_bias_ttnn, dtype=torch.float32) if conv_bias_ttnn is not None else None
    bn_weight = bn_weight_ttnn.squeeze()  # Remove batch/spatial dims if present
    bn_bias = bn_bias_ttnn.squeeze()
    bn_running_mean = bn_running_mean_ttnn.squeeze()
    bn_running_var = bn_running_var_ttnn.squeeze()

    # Ensure all tensors are on the same device and dtype
    device = conv_weight.device
    dtype = conv_weight.dtype

    # Move all tensors to the same device if needed
    bn_weight = bn_weight.to(device=device, dtype=dtype)
    bn_bias = bn_bias.to(device=device, dtype=dtype)
    bn_running_mean = bn_running_mean.to(device=device, dtype=dtype)
    bn_running_var = bn_running_var.to(device=device, dtype=dtype)

    # Handle case where conv has no bias
    if conv_bias is None:
        logger.debug("Convolution has no bias; initializing to zeros for fusion.")
        conv_bias = torch.zeros(conv_weight.shape[0], device=device, dtype=dtype)
    else:
        conv_bias = conv_bias.to(device=device, dtype=dtype)

    # Compute the scaling factor: bn_weight / sqrt(bn_running_var + eps)
    scale = bn_weight / torch.sqrt(bn_running_var + eps)

    # Reshape scale to match the dimensions of the convolutional weights [out_channels, 1, 1, 1]
    scale_reshaped = scale.view(-1, 1, 1, 1)

    # Scale the convolutional weights
    fused_weight = conv_weight * scale_reshaped

    # Compute the fused bias: (conv_bias - bn_running_mean) * scale + bn_bias
    # fused_bias = (conv_bias - bn_running_mean) * scale + bn_bias
    fused_bias = (conv_bias - bn_running_mean) * scale + bn_bias

    # Convert back to TTNN tensors
    # Weights stay on host for now
    fused_weight_ttnn = ttnn.from_torch(fused_weight, dtype=ttnn.bfloat16)

    # Handle bias tensor shape for TTNN (reshape to [1, 1, 1, -1] if needed)
    if len(fused_bias.shape) == 1:
        fused_bias = fused_bias.reshape((1, 1, 1, -1))
    fused_bias_ttnn = ttnn.from_torch(fused_bias, device=conv_weight_ttnn.device(), dtype=ttnn.bfloat16)

    return fused_weight_ttnn, fused_bias_ttnn


def fuse_conv_bn_parameters(parameters, eps=1e-5):
    """
    Fuse Conv+BatchNorm patterns in preprocessed parameters.

    This function takes the parameters object returned by create_panoptic_deeplab_parameters()
    and performs Conv+BN fusion, returning a new parameters object with only the fused
    convolution weights.

    Args:
        parameters: Parameter dict returned by create_panoptic_deeplab_parameters()
        eps (float): BatchNorm epsilon value for numerical stability

    Returns:
        dict: New parameter dict with fused Conv weights and biases
    """

    def process_module(module_params, path=""):
        """Recursively process parameters and fuse Conv+BN patterns."""
        fused_params = module_params.copy()

        for key, value in fused_params.items():
            current_path = f"{path}.{key}" if path else key
            # logger.debug(f"Processing parameter at: {current_path}")

            conv_bn_keys = [('conv1', 'bn1'), ('conv2', 'bn2'), ('lat8', 'bn8'), ('lat16', 'bn16'), ('lat32', 'bn32')]

            if isinstance(value, dict):
                # Check if this is a Conv+BN pattern (has both 'conv' and 'bn' keys)
                for conv_key, bn_key in conv_bn_keys:
                    if conv_key in value and bn_key in value:
                        logger.debug(f"Fusing Conv+BN parameters at: {current_path} for keys {conv_key}, {bn_key}")
                        # Extract conv parameters (TTNN tensors)
                        conv_params = value[conv_key]
                        conv_weight = conv_params["weight"]
                        conv_bias = conv_params.get("bias", None)

                        # Extract norm parameters (TTNN tensors)
                        norm_params = value[bn_key]
                        bn_weight = norm_params["weight"]
                        bn_bias = norm_params["bias"]
                        bn_running_mean = norm_params["running_mean"]
                        bn_running_var = norm_params["running_var"]

                        # Perform fusion
                        fused_weight, fused_bias = fuse_conv_bn_weights_ttnn(
                            conv_weight_ttnn=conv_weight,
                            conv_bias_ttnn=conv_bias,
                            bn_weight_ttnn=bn_weight,
                            bn_bias_ttnn=bn_bias,
                            bn_running_mean_ttnn=bn_running_mean,
                            bn_running_var_ttnn=bn_running_var,
                            eps=eps,
                        )
                        value[conv_key]["weight"] = fused_weight
                        value[conv_key]["bias"] = fused_bias
                        

                conv_key = 0
                bn_key = 1

                if conv_key in value and bn_key in value and key == 'downsample':
                        logger.debug(f"Fusing Conv+BN parameters at: {current_path} for keys {conv_key}, {bn_key}")
                        # Extract conv parameters (TTNN tensors)
                        conv_params = value[conv_key]
                        conv_weight = conv_params["weight"]
                        conv_bias = conv_params.get("bias", None)

                        # Extract norm parameters (TTNN tensors)
                        norm_params = value[bn_key]
                        bn_weight = norm_params["weight"]
                        bn_bias = norm_params["bias"]
                        bn_running_mean = norm_params["running_mean"]
                        bn_running_var = norm_params["running_var"]

                        # Perform fusion
                        fused_weight, fused_bias = fuse_conv_bn_weights_ttnn(
                            conv_weight_ttnn=conv_weight,
                            conv_bias_ttnn=conv_bias,
                            bn_weight_ttnn=bn_weight,
                            bn_bias_ttnn=bn_bias,
                            bn_running_mean_ttnn=bn_running_mean,
                            bn_running_var_ttnn=bn_running_var,
                            eps=eps,
                        )
                        value[conv_key]["weight"] = fused_weight
                        value[conv_key]["bias"] = fused_bias

                process_module(value, current_path)

        return fused_params

    return process_module(parameters)


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
    if isinstance(model, torch.nn.Linear):
        parameters[f"weight"] = preprocess_linear_weight(model.weight, dtype=ttnn.bfloat16)
        if model.bias is not None:
            parameters[f"bias"] = preprocess_linear_bias(model.bias, dtype=ttnn.bfloat16)
    if isinstance(model, nn.GroupNorm):
        parameters["weight"] = model.weight
        if model.bias is not None:
            parameters["bias"] = model.bias
    if isinstance(model, nn.BatchNorm2d):
        parameters["weight"] = model.weight
        if model.bias is not None:
            parameters["bias"] = model.bias
        parameters["running_mean"] = model.running_mean
        parameters["running_var"] = model.running_var

    return parameters


def create_OFT_model_parameters_resnet(model, input_tensor: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )
    parameters.layer_args = make_parameter_dict({})
    parameters.layer_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(input_tensor), device=None
    )
    parameters.model = model

    return parameters


def create_OFT_model_parameters_oft(model, input_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.layer_args = make_parameter_dict({})
    parameters.layer_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(*input_tensors),
        device=None,
    )

    parameters["model_args"] = model

    return parameters


def create_OFT_model_parameters(model, input_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )
    if model.oft8 is not None:
        parameters.oft8.conv3d.weight = ttnn.to_device(parameters.oft8.conv3d.weight, device=device)
        parameters.oft8.conv3d.bias = ttnn.to_device(parameters.oft8.conv3d.bias, device=device)
    if model.oft16 is not None:
        parameters.oft16.conv3d.weight = ttnn.to_device(parameters.oft16.conv3d.weight, device=device)
        parameters.oft16.conv3d.bias = ttnn.to_device(parameters.oft16.conv3d.bias, device=device)
    if model.oft32 is not None:
        parameters.oft32.conv3d.weight = ttnn.to_device(parameters.oft32.conv3d.weight, device=device)
        parameters.oft32.conv3d.bias = ttnn.to_device(parameters.oft32.conv3d.bias, device=device)

    parameters.layer_args = make_parameter_dict({})
    parameters.layer_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(*input_tensors),
        device=None,
    )

    parameters["model_args"] = model

    return parameters


def create_decoder_model_parameters(model, input_tensors: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )

    parameters.layer_args = make_parameter_dict({})
    parameters.layer_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model.decode(*input_tensors), device=None
    )

    parameters["model_args"] = model

    return parameters
