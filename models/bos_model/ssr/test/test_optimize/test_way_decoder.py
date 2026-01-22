import argparse
from test.common import MyDict
from test.utils import compare_tensors, pt2tt

import pytest
import torch
from bos_metal import device_box, engine, op
from mmcv.cnn.bricks.registry import ATTENTION, NORM_LAYERS, TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

import ttnn

NORM_LAYERS.register_module("LN_tt", module=op.LayerNorm, force=True)


def random_inputs(device):
    """Generate deterministic inputs and their TTNN copies."""
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 18
    embed_dims = 256

    ref_query = torch.randn(seq_len, batch_size, embed_dims)
    query = pt2tt(ref_query.permute(1, 0, 2), memory_config=ttnn.L1_MEMORY_CONFIG, device=device)
    ref_key = torch.randn(seq_len, batch_size, embed_dims)
    key = pt2tt(ref_key.permute(1, 0, 2), memory_config=ttnn.L1_MEMORY_CONFIG, device=device)
    ref_value = torch.randn(seq_len, batch_size, embed_dims)
    value = pt2tt(ref_value.permute(1, 0, 2), memory_config=ttnn.L1_MEMORY_CONFIG, device=device)
    # ------------------
    ref_query_pos = torch.randn(seq_len, 1, embed_dims)
    query_pos = pt2tt(ref_query_pos.permute(1, 0, 2), memory_config=ttnn.L1_MEMORY_CONFIG, device=device)
    ref_key_pos = torch.randn(seq_len, 1, embed_dims)
    key_pos = pt2tt(ref_key_pos.permute(1, 0, 2), memory_config=ttnn.L1_MEMORY_CONFIG, device=device)

    ref_input = {
        "query": ref_query,
        "key": ref_key,
        "value": ref_value,
        "query_pos": ref_query_pos,
        "key_pos": ref_key_pos,
    }
    input = {
        "query": query,
        "key": key,
        "value": value,
        "query_pos": query_pos,
        "key_pos": key_pos,
    }

    return ref_input, input


memory_config = MyDict(
    {
        "cross_attn": {
            "q_input": ttnn.L1_MEMORY_CONFIG,
            "kv_input": ttnn.L1_MEMORY_CONFIG,
            "in_proj_q": ttnn.L1_MEMORY_CONFIG,
            "in_proj_kv": ttnn.L1_MEMORY_CONFIG,
            "qkv_split": ttnn.L1_MEMORY_CONFIG,
            "out_proj": ttnn.L1_MEMORY_CONFIG,
            "attn_scores": ttnn.L1_MEMORY_CONFIG,
            "attn_softmax": ttnn.L1_MEMORY_CONFIG,
            "qkv_permute": ttnn.L1_MEMORY_CONFIG,
        },
        "norm": ttnn.L1_MEMORY_CONFIG,
        "ffn": ttnn.L1_MEMORY_CONFIG,
    }
)

way_decoder=dict(
        type='CustomTransformerDecoder',
        num_layers=1,
        return_intermediate=False,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8),
            ],
            feedforward_channels=512,
            # ffn_dropout=0.1,
            operation_order=('cross_attn', 'norm', 'ffn', 'norm')))

def run_with_pcc(device):
    ref_input, input = random_inputs(device)

    ref_model = build_transformer_layer_sequence(way_decoder)

    from test.common import BaseTransformerLayer

    ATTENTION.register_module(op.MultiheadAttention, force=True)
    TRANSFORMER_LAYER.register_module(BaseTransformerLayer, force=True)
    model = build_transformer_layer_sequence(way_decoder)

    # Load processed state dict into TTNN model
    state_dict = engine.ModelProcessor(ref_model).process_state_dict(**ref_input)
    model.load_state_dict(state_dict, strict=False)

    # Forward passes
    ref_out = ref_model(**ref_input).permute(1, 0, 2)
    out = model(**input, memory_config=memory_config)

    compare_tensors(ref_out, out)


def run_with_speed(device, num_run):
    ref_input, input = random_inputs(device)

    ref_model = build_transformer_layer_sequence(way_decoder)

    from test.common import BaseTransformerLayer

    ATTENTION.register_module(op.MultiheadAttention, force=True)
    TRANSFORMER_LAYER.register_module(BaseTransformerLayer, force=True)
    model = build_transformer_layer_sequence(way_decoder)

    # Load processed state dict into TTNN model
    state_dict = engine.ModelProcessor(ref_model).process_state_dict(**ref_input)
    model.load_state_dict(state_dict, strict=False)

    for i in range(num_run + 1):
        if i == 0:
            out = model(**input, memory_config=memory_config)
            ttnn.synchronize_device(device)
        else:
            with op.time_profiler(verbose=False):
                out = model(**input, memory_config=memory_config)
                ttnn.synchronize_device(device)
        if i != 10:
            ttnn.deallocate(out)
    op.timer.print_report(units="ms")


def parse_args():
    parser = argparse.ArgumentParser(description="Way Decoder Test Script")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcc", action="store_true", help="Enable PCC mode")
    group.add_argument("--speed", action="store_true", help="Enable speed optimization")
    return parser.parse_args()

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
