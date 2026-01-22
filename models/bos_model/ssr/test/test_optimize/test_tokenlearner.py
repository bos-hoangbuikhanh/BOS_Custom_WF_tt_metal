import argparse
import math
from test.common import GELU, MyDict
from test.SSR.reference.tokenlearner import TokenLearnerV11_Torch
from test.utils import compare_tensors, pt2tt
from test.SSR.tt.tokenlearner import TokenLearnerV11
from test.configs.op_configs import memory_config, program_config
import pytest
import torch
from bos_metal import device_box, op

import ttnn


def random_inputs(device):
    """Generate deterministic inputs and their TTNN copies."""
    torch.manual_seed(0)

    input_torch = torch.randn(1, 1, 10_000, 512)
    input_tt = pt2tt(input_torch, device=device)
    input_torch = input_torch.permute(0, 3, 1, 2)  # (1, 512, 1, 10_000)

    return input_torch, input_tt


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenlearner Test Script")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcc", action="store_true", help="Enable PCC mode")
    group.add_argument("--speed", action="store_true", help="Enable speed optimization")
    return parser.parse_args()

def run_with_pcc(device):
    ref_x, x = random_inputs(device)
    tokenlearner_torch = TokenLearnerV11_Torch(16, 512)
    tokenlearner_torch.eval()
    # with torch.no_grad():
    #     tokenlearner_torch.layer_norm.weight.copy_(
    #         torch.rand_like(tokenlearner_torch.layer_norm.weight).to(torch.bfloat16)
    #     )
    #     tokenlearner_torch.layer_norm.bias.copy_(
    #         torch.rand_like(tokenlearner_torch.layer_norm.bias).to(torch.bfloat16)
    #     )
    state_dict = tokenlearner_torch.state_dict()
    tokenlearner_tt = TokenLearnerV11(
        num_tokens=16, in_channels=512, bottleneck_dim=64, num_out_blocks=16, dropout_rate=0.0, device=device_box.get()
    )
    tokenlearner_tt.load_state_dict(state_dict)
    ref_out, ref_selected = tokenlearner_torch(ref_x)
    out = tokenlearner_tt(x, memory_config=memory_config['SSRHead']['tokenlearner'], program_config=program_config['SSRHead']['tokenlearner'])
    compare_tensors(ref_out, out)


def run_with_speed(device, num_run):
    ref_x, x = random_inputs(device)
    tokenlearner_torch = TokenLearnerV11_Torch(16, 512)
    tokenlearner_torch.eval()
    state_dict = tokenlearner_torch.state_dict()
    tokenlearner_tt = TokenLearnerV11(
        num_tokens=16, in_channels=512, bottleneck_dim=64, num_out_blocks=16, dropout_rate=0.0, device=device_box.get()
    )
    tokenlearner_tt.load_state_dict(state_dict)

    for i in range(num_run + 1):
        if i == 0:
            out = tokenlearner_tt(x, memory_config=memory_config['SSRHead']['tokenlearner'], program_config=program_config['SSRHead']['tokenlearner'])
            ttnn.synchronize_device(device)
        else:
            with op.time_profiler(verbose=False):
                out = tokenlearner_tt(x, memory_config=memory_config, program_config=program_config['SSRHead']['tokenlearner'])
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
