import os
import time
from test.SSR.tt.fpn import FPN

import pytest
import torch
from bos_metal import compare_tensors, device_box, engine, op, ttnn
from mmdet.models.builder import BACKBONES, build_backbone
from tt.projects.configs import fpn


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def device():
    """Open Tenstorrent device once per pytest session and close on teardown."""
    device = device_box.open({"device_id": 0, "l1_small_size": 64 * 1024}, enable_program_cache=True)
    yield device
    device_box.close()


def test_fpn(device):
    os.unsetenv("TTNN_ENABLE_LOGGING")
    os.unsetenv("ENABLE_PROFILER")
    os.unsetenv("TT_METAL_DPRINT_CORES")
    os.unsetenv("TT_METAL_DPRINT_CHIPS")
    os.unsetenv("ENABLE_TRACY")
    os.unsetenv("TT_METAL_DEVICE_PROFILER")
    os.unsetenv("TT_METAL_PROFILER_SYNC")
    os.unsetenv("TT_METAL_DEVICE_PROFILER_DISPATCH")

    # 1. Initialize the TT-NN device
    # ttnn.device.EnablePersistentKernelCache()

    # 2. Define parameters
    model_name = "FPN"
    in_channels = [2048]
    out_channels = 256
    start_level = 0
    batch_size = 6
    input_height = [12]  # Example input heights for each channel
    input_width = [20]  # Example input widths for each channel

    # 3. Initialize Pytorch and TT-NN models
    ## 3.1. Define ResNet50 backbone configuration
    img_backbone = dict(
        type="FPN",
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=start_level,
        add_extra_convs="on_output",
        num_outs=1,
        relu_before_extra_convs=True,
    )

    ## 3.2. Initialize data
    torch_input_tensor = [
        torch.randn((batch_size, in_channels[i], input_height[i], input_width[i])) for i in range(len(in_channels))
    ]

    ## 3.3. Initialize Pytorch model
    torch_fpn = build_backbone(img_backbone)
    torch_fpn.eval()

    ## 3.4. Initialize TT-NN model
    BACKBONES.register_module(f"{model_name}_tt", module=FPN, force=True)
    img_backbone["type"] = f"{model_name}_tt"
    ttnn_fpn = build_backbone(img_backbone)

    ### 3.4.1. Load parameters
    processor = engine.ModelProcessor(model=torch_fpn)
    state_dict = processor.process_state_dict(torch_input_tensor)
    for key in [
        "input_shape",
        "output_shape",
        "lateral_convs.0.input_shape",
        "lateral_convs.0.output_shape",
        "lateral_convs.1.input_shape",
        "lateral_convs.1.output_shape",
        "lateral_convs.2.input_shape",
        "lateral_convs.2.output_shape",
        "lateral_convs.3.input_shape",
        "lateral_convs.3.output_shape",
        "fpn_convs.0.input_shape",
        "fpn_convs.0.output_shape",
        "fpn_convs.1.input_shape",
        "fpn_convs.1.output_shape",
        "fpn_convs.2.input_shape",
        "fpn_convs.2.output_shape",
        "fpn_convs.3.input_shape",
        "fpn_convs.3.output_shape",
    ]:
        if key in state_dict:
            del state_dict[key]

    msg = ttnn_fpn.load_state_dict(state_dict)  # PyTorch inherited function
    print("=" * 100)
    print(msg)
    print("=" * 100)

    ### 3.4.2. Load config
    msg = ttnn_fpn.load_config_dict(fpn.module_config)
    print("=" * 100)
    print(msg)
    print("=" * 100)

    # 4. Run model
    ## 4.1. Pytorch model
    torch_output_tensors = torch_fpn(torch_input_tensor)

    ## 4.2. TT-NN model
    ttnn_input_tensor = [torch.permute(tensor, (0, 2, 3, 1)) for tensor in torch_input_tensor]
    ttnn_input_tensor = [
        ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
        for tensor in ttnn_input_tensor
    ]

    # Warmup runs
    ttnn_output_tensor = ttnn_fpn(ttnn_input_tensor)
    for i in range(len(ttnn_output_tensor)):
        temp_out = ttnn.to_torch(ttnn_output_tensor[i])
        ttnn.deallocate(ttnn_output_tensor[i])
        del temp_out

    # I don't know why input tensor is deallocated here, so re-create it
    ttnn_input_tensor = [torch.permute(tensor, (0, 2, 3, 1)) for tensor in torch_input_tensor]
    ttnn_input_tensor = [
        ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
        for tensor in ttnn_input_tensor
    ]
    # Actual runs

    with op.time_profiler():
        op.timer.start("ttnn fpn")
        start = time.time()
        ttnn_output_tensor = ttnn_fpn(ttnn_input_tensor)
        elapse_time = time.time() - start
        print(f"TT-NN FPN forward pass took {elapse_time:.4f} seconds")
        op.timer.end("ttnn fpn")
        op.timer.print_report(units="ms")
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor[-1])
    torch_output_shape = torch_output_tensors[-1].shape
    ttnn_output_tensor = torch.reshape(
        ttnn_output_tensor, (torch_output_shape[0], torch_output_shape[2], torch_output_shape[3], torch_output_shape[1])
    )
    ttnn_output_tensor = torch.permute(ttnn_output_tensor, (0, 3, 1, 2))  # Change back to (N, C, H, W) format
    ttnn_output_tensor = ttnn_output_tensor.to(
        torch_output_tensors[-1].dtype
    )  # Convert to the same dtype as input tensor

    # 5. Compare the outputs
    print("=" * 100)
    compare_tensors(torch_output_tensors[-1], ttnn_output_tensor)
    print("=" * 100)

    # device_box.close()
