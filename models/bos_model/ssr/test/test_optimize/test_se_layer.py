import argparse
import math
from test.common import MyDict
from test.SSR.reference.squeeze_excitation import SELayer as SELayerV1
from test.SSR.tt.squeeze_excitation import SELayer as SELayerV2
from test.utils import compare_tensors, pt2tt

import pytest
import torch
from bos_metal import device_box, engine, op

import ttnn


def random_inputs(device):
    """Generate deterministic inputs and their TTNN copies."""
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 10_000
    num_channels = 256

    ref_x = torch.randn(batch_size, seq_len, num_channels)
    ref_x_se = torch.randn(1, 1, num_channels)

    x = pt2tt(ref_x, memory_config=ttnn.L1_MEMORY_CONFIG, device=device)

    x_se = pt2tt(ref_x_se, memory_config=ttnn.L1_MEMORY_CONFIG, device=device)
    return num_channels, ref_x, ref_x_se, x, x_se


mu_reuse_cast_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(5, 4),
    in0_block_w=1,
    out_subblock_h=1,
    out_subblock_w=1,
    out_block_h=math.ceil(10000 / 32 / 5),
    out_block_w=math.ceil(256 / 32 / 4),
    per_core_M=math.ceil(10000 / 32 / 5),
    per_core_N=math.ceil(256 / 32 / 4),
    transpose_mcast=True,  # transpose for col only
    fuse_batch=True,
    fused_activation=None,
)

core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))])

memory_config_x = ttnn.create_sharded_memory_config(
    shape=[64, 2016],
    core_grid=core_grid,
    strategy=ttnn.ShardStrategy.BLOCK,
    use_height_and_width_as_shard_shape=True,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
)

program_config = MyDict(
    {
        "mlp_reduce": mu_reuse_cast_program_config,
        "mlp_expand": mu_reuse_cast_program_config,
    }
)

memory_config = MyDict(
    {
        "x": memory_config_x,
        "mlp_reduce": ttnn.L1_MEMORY_CONFIG,
        "mlp_expand": ttnn.L1_MEMORY_CONFIG,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="SE Layer Test Script")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcc", action="store_true", help="Enable PCC mode")
    group.add_argument("--speed", action="store_true", help="Enable speed optimization")
    return parser.parse_args()

def run_with_pcc(device):
    num_channels, ref_x, ref_x_se, x, x_se = random_inputs(device)
    ref_model = SELayerV1(num_channels)
    model = SELayerV2(num_channels)
    state_dict = engine.ModelProcessor(ref_model).process_state_dict(x=ref_x, x_se=ref_x_se)
    model.load_state_dict(state_dict, strict=False)
    ref_out = ref_model(ref_x, ref_x_se)  # torch version

    x = ttnn.to_memory_config(x, memory_config=memory_config_x)
    x = ttnn.unsqueeze(x, 0)
    out = model(x, x_se, memory_config=memory_config, program_config=program_config)
    ttnn.synchronize_device(device)
    compare_tensors(ref_out, out)
    assert compare_tensors(ref_out, out)


def run_with_speed(device, num_run):
    num_channels, ref_x, ref_x_se, x, x_se = random_inputs(device)
    ref_model = SELayerV1(num_channels)
    model = SELayerV2(num_channels)
    state_dict = engine.ModelProcessor(ref_model).process_state_dict(x=ref_x, x_se=ref_x_se)
    model.load_state_dict(state_dict, strict=False)
    ref_out = ref_model(ref_x, ref_x_se)  # torch version

    for i in range(num_run + 1):
        if i == 0:
            x = ttnn.to_memory_config(x, memory_config=memory_config_x)
            x = ttnn.unsqueeze(x, 0)
            out = model(x, x_se, memory_config=memory_config, program_config=program_config)
            ttnn.synchronize_device(device)
        else:
            with op.time_profiler(verbose=False):
                x = ttnn.to_memory_config(x, memory_config=memory_config_x)
                x = ttnn.unsqueeze(x, 0)
                out = model(x, x_se, memory_config=memory_config, program_config=program_config)
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
