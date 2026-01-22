# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import torch.nn as nn
from ...reference.resnet import BasicBlock
from ...tt.model_configs import ModelOptimizations
from ...tt.tt_resnet import TTBasicBlock
from tests.ttnn.utils_for_testing import assert_with_pcc

# from ttnn.model_preprocessing import preprocess_model_parameters
from ...tt.model_preprocessing import create_OFT_model_parameters_resnet, fuse_conv_bn_parameters

try:
    from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
except ImportError:
    from tests.ttnn.unit_tests.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores

from loguru import logger


@pytest.mark.parametrize(
    "n, in_ch, out_ch, h, w, stride, to_dram",
    [(1, 128, 128, 159, 159, 1, True)],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8 * 1024}], indirect=True)
def test_tt_topdown_network(device, n, in_ch, out_ch, h, w, stride, to_dram):
    skip_if_not_blackhole_20_cores(device)
    device.disable_and_clear_program_cache()  # test hangs without this line on P150
    torch.manual_seed(42)

    input_tensor = torch.randn(n, in_ch, h, w)

    # Create a sequence of 8 BasicBlocks to simulate a topdown network
    torch_model = nn.Sequential(*[BasicBlock(inplanes=in_ch, planes=out_ch) for _ in range(8)])
    torch_model.eval()

    state_dict = create_OFT_model_parameters_resnet(torch_model, input_tensor, device)
    state_dict = {'topdown': state_dict}
    state_dict = fuse_conv_bn_parameters(state_dict)['topdown']

    # Get reference output
    out_ref = torch_model(input_tensor)

    # Apply model optimizations
    model_opt = ModelOptimizations()
    model_opt.apply(state_dict, "topdown")

    tt_blocks = [
        TTBasicBlock(
            device,
            state_dict[i],
            state_dict.layer_args[i],
        )
        for i in range(8)
    ]
    # Prepare TTNN input
    n, c, h, w = input_tensor.shape
    x_for_ttnn = input_tensor.permute(0, 2, 3, 1).view(1, 1, n * h * w, c)
    ttnn_x = ttnn.from_torch(x_for_ttnn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # Forward through TTBasicBlocks sequentially
    ttnn_out = ttnn_x
    for block in tt_blocks:
        ttnn_out = block.forward(device, ttnn_out, move_to_dram=to_dram)
    ttnn_out = ttnn.to_torch(ttnn_out)
    # Compare output
    B, C, H, W = out_ref.shape
    out_ref = out_ref.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C)
    pcc, message = assert_with_pcc(ttnn_out, out_ref, 0.998)
    logger.info(f"PCC for topdown block with 8 BasicBlocks: {pcc}, Message: {message}")


@pytest.mark.parametrize(
    "n, in_ch, out_ch, h, w, stride, to_dram",
    [
        (1, 128, 128, 48, 160, 1, True),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8 * 1024}], indirect=True)
def test_tt_basicblock_single(device, n, in_ch, out_ch, h, w, stride, to_dram):
    skip_if_not_blackhole_20_cores(device)
    torch.manual_seed(42)
    input_tensor = torch.randn(n, in_ch, h, w)
    torch_model = BasicBlock(inplanes=in_ch, planes=out_ch, stride=stride)
    torch_model.eval()
    out = torch_model.forward(input_tensor)
    
    state_dict = create_OFT_model_parameters_resnet(torch_model, input_tensor, device)
    state_dict = {'topdown.0': state_dict}
    state_dict = fuse_conv_bn_parameters(state_dict)['topdown.0']

    # Apply model optimizations
    model_opt = ModelOptimizations()
    model_opt.apply(state_dict, "topdown.0")  # to update path to real one

    block = TTBasicBlock(device, state_dict, state_dict.layer_args)

    n, c, h, w = input_tensor.shape
    x_for_ttnn = input_tensor.permute(0, 2, 3, 1).view(1, 1, n * h * w, c)
    ttnn_x = ttnn.from_torch(x_for_ttnn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_out = block.forward(device, ttnn_x, move_to_dram=to_dram)

    logger.info(f"Output shape: {ttnn_out.shape}, torch out {out.shape}")
    B, C, H, W = out.shape
    ttnn_out = ttnn.to_torch(ttnn_out)
    out = out.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C)
    pcc, message = assert_with_pcc(ttnn_out, out, 0.999)
    logger.info(f"PCC: {pcc}, Message: {message}")
