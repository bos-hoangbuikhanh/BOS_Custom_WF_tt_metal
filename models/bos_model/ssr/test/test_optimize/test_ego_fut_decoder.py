import argparse
from test.common import MyDict, ReLU
from test.utils import compare_tensors, pt2tt

import pytest
import torch
from bos_metal import device_box, engine, op

import ttnn


def random_inputs(device):
    """Generate deterministic inputs and their TTNN copies."""
    torch.manual_seed(0)

    batch_size = 1
    num_channels = 256
    num_reg_fcs = 2

    ref_x = torch.randn(batch_size, 18, num_channels)
    x = pt2tt(ref_x, memory_config=ttnn.L1_MEMORY_CONFIG, device=device)

    return ref_x, x, num_reg_fcs


memory_config = MyDict(
    {
        "linear1": ttnn.L1_MEMORY_CONFIG,
        "linear2": ttnn.L1_MEMORY_CONFIG,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="SE Layer Test Script")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcc", action="store_true", help="Enable PCC mode")
    group.add_argument("--speed", action="store_true", help="Enable speed optimization")
    return parser.parse_args()


def create_model(num_reg_fcs, Linear, ReLU, embed_dims=256):
    ego_fut_decoder = []
    for _ in range(num_reg_fcs):
        # TODO: Fuse Linear and ReLU
        ego_fut_decoder.append(Linear(embed_dims, embed_dims))
        ego_fut_decoder.append(ReLU())
    ego_fut_decoder.append(Linear(embed_dims, 2))

    return torch.nn.Sequential(*ego_fut_decoder)

def run_with_pcc(device):
    ref_input, input, num_reg_fcs = random_inputs(device)
    ref_model = create_model(num_reg_fcs, torch.nn.Linear, torch.nn.ReLU)
    model = create_model(num_reg_fcs, op.Linear, ReLU)

    # Load processed state dict into TTNN model
    state_dict = engine.ModelProcessor(ref_model).process_state_dict(ref_input)
    model.load_state_dict(state_dict, strict=False)

    ref_out = ref_model(ref_input)
    out = model(input)
    compare_tensors(ref_out, out)
    assert compare_tensors(ref_out, out)


def run_with_speed(device, num_run):
    ref_input, input, num_reg_fcs = random_inputs(device)
    ref_model = create_model(num_reg_fcs, torch.nn.Linear, torch.nn.ReLU)
    model = create_model(num_reg_fcs, op.Linear, ReLU)

    # Load processed state dict into TTNN model
    state_dict = engine.ModelProcessor(ref_model).process_state_dict(ref_input)
    model.load_state_dict(state_dict, strict=False)

    for i in range(num_run + 1):
        if i == 0:
            out = model(input)
            ttnn.synchronize_device(device)
        else:
            with op.time_profiler(verbose=False):
                out = model(input)
                ttnn.synchronize_device(device)
        if i != 10:
            ttnn.deallocate(out)
    op.timer.print_report(units="ms")

@pytest.mark.parametrize("device", [device_box.open(enable_program_cache=False)])
def test_pcc(device):
    run_with_pcc(device)


if __name__ == "__main__":
    device = device_box.open(enable_program_cache=False)
    args = parse_args()
    if args.pcc:
        print("Run with PCC check")
        run_with_pcc(device)

    if args.speed:
        print("Run with speed check")
        run_with_speed(device, num_run=10)
