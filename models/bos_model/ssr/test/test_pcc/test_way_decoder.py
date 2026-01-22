import pytest
import torch
import ttnn

from bos_metal import op, device_box, engine
from test.utils import pt2tt, compare_tensors
from test.common import MyDict

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, ATTENTION, NORM_LAYERS
NORM_LAYERS.register_module('LN_tt', module=op.LayerNorm, force=True)

# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def device():
    """Open a TT device once per test session and return it."""
    device_box.open()
    dev = device_box.get()
    yield dev
    device_box.close()


def random_inputs(device):
    """Generate deterministic inputs and their TTNN copies."""
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 18
    embed_dims = 256

    ref_query = torch.randn(seq_len, batch_size, embed_dims)
    query = pt2tt(ref_query.permute(1, 0, 2),
              memory_config=ttnn.L1_MEMORY_CONFIG,
              device=device)
    ref_key = torch.randn(seq_len, batch_size, embed_dims)
    key = pt2tt(ref_key.permute(1, 0, 2),
              memory_config=ttnn.L1_MEMORY_CONFIG,
              device=device)
    ref_value = torch.randn(seq_len, batch_size, embed_dims)
    value = pt2tt(ref_value.permute(1, 0, 2),
              memory_config=ttnn.L1_MEMORY_CONFIG,
              device=device)
    # ------------------
    ref_query_pos = torch.randn(seq_len, 1, embed_dims)
    query_pos = pt2tt(ref_query_pos.permute(1, 0, 2),
              memory_config=ttnn.L1_MEMORY_CONFIG,
              device=device)
    ref_key_pos = torch.randn(seq_len, 1, embed_dims)
    key_pos = pt2tt(ref_key_pos.permute(1, 0, 2),
              memory_config=ttnn.L1_MEMORY_CONFIG,
              device=device)

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

# --------------------------------------------------------------------------- #
#  Test                                                                        #
# --------------------------------------------------------------------------- #

def test_way_decoder_equivalence(device):
    ref_input, input = random_inputs(device)

    # Layer config
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
    
    # Initialise both models
    from test.SSR.reference.custom_transformer_decoder import CustomTransformerDecoder
    ref_model = build_transformer_layer_sequence(way_decoder)
    
    from test.SSR.tt.custom_transformer_decoder import CustomTransformerDecoder
    from test.common import BaseTransformerLayer
    ATTENTION.register_module(op.MultiheadAttention, force=True)
    TRANSFORMER_LAYER.register_module(BaseTransformerLayer, force=True)
    model = build_transformer_layer_sequence(way_decoder)

    # Load processed state dict into TTNN model
    state_dict = engine.ModelProcessor(ref_model)\
                       .process_state_dict(**ref_input)
    model.load_state_dict(state_dict, strict=False)

    # Forward passes
    ref_out = ref_model(**ref_input).permute(1, 0, 2)
    out = model(**input)
    
    # Assertions
    assert ref_out.shape == out.shape, "Output shapes do not match"
    assert compare_tensors(ref_out, out)[0], "Tensor values differ beyond tolerance"

