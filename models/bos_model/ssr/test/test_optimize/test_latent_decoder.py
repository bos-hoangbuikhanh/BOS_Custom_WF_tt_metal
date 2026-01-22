import pytest
import argparse 
import torch
import ttnn
import math

from bos_metal import op, device_box, engine
from test.utils import pt2tt, compare_tensors
from test.common import MyDict

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, ATTENTION, NORM_LAYERS
NORM_LAYERS.register_module('LN_tt', module=op.LayerNorm, force=True)

def random_inputs(device):
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 16
    embed_dims = 256

    ref_query = torch.randn(seq_len, batch_size, embed_dims)
    query = pt2tt(ref_query.permute(1, 0, 2),
              memory_config=ttnn.L1_MEMORY_CONFIG,
              device=device)
    ref_query_pos = torch.randn(seq_len, 1, embed_dims)
    query_pos = pt2tt(ref_query_pos.permute(1, 0, 2),
              memory_config=ttnn.L1_MEMORY_CONFIG,
              device=device)

    ref_input = {
        "query": ref_query,
        "key": ref_query,
        "value": ref_query,
        "query_pos": ref_query_pos,
        "key_pos": ref_query_pos,
    }
    input = {
        "query": query,
        "key": query,
        "value": query,
        "query_pos": query_pos,
        "key_pos": query_pos,
    }
    return ref_input, input

memory_config = MyDict(
    {
        "self_attn": 
        {
            "q_input": ttnn.L1_MEMORY_CONFIG,
            "kv_input": ttnn.L1_MEMORY_CONFIG,
            "in_proj_q": ttnn.L1_MEMORY_CONFIG,
            "in_proj_kv": ttnn.L1_MEMORY_CONFIG,
            "qkv_permute": ttnn.L1_MEMORY_CONFIG,
            "qkv_split": ttnn.L1_MEMORY_CONFIG,
        },
        "norm": ttnn.L1_MEMORY_CONFIG,
        "ffn": ttnn.L1_MEMORY_CONFIG,

    }
)

latent_decoder=dict(
        type='CustomTransformerDecoder',
        num_layers=3,
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
            operation_order=('self_attn', 'norm', 'ffn', 'norm')))

def run_with_pcc(device):
    ref_input, input = random_inputs(device)
    from test.SSR.reference.custom_transformer_decoder import CustomTransformerDecoder
    ref_model = build_transformer_layer_sequence(latent_decoder)
    
    from test.SSR.tt.custom_transformer_decoder import CustomTransformerDecoder
    from test.common import BaseTransformerLayer
    ATTENTION.register_module(op.MultiheadAttention, force=True)
    TRANSFORMER_LAYER.register_module(BaseTransformerLayer, force=True)
    model = build_transformer_layer_sequence(latent_decoder)

    # Load processed state dict into TTNN model
    state_dict = engine.ModelProcessor(ref_model)\
                       .process_state_dict(**ref_input)
    model.load_state_dict(state_dict, strict=False)

    # Forward passes
    ref_out = ref_model(**ref_input).permute(1, 0, 2)
    out = model(**input, memory_config=memory_config)

    compare_tensors(ref_out, out)

def run_with_speed(device, num_run):
    ref_input, input = random_inputs(device)
    from test.SSR.reference.custom_transformer_decoder import CustomTransformerDecoder
    ref_model = build_transformer_layer_sequence(latent_decoder)
    
    from test.SSR.tt.custom_transformer_decoder import CustomTransformerDecoder
    from test.common import BaseTransformerLayer
    ATTENTION.register_module(op.MultiheadAttention, force=True)
    TRANSFORMER_LAYER.register_module(BaseTransformerLayer, force=True)
    model = build_transformer_layer_sequence(latent_decoder)

    # Load processed state dict into TTNN model
    state_dict = engine.ModelProcessor(ref_model)\
                       .process_state_dict(**ref_input)
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
    parser = argparse.ArgumentParser(description="Latent Decoder Test Script")
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