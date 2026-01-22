import pytest
import torch
import ttnn

from bos_metal import op, device_box, engine
from test.utils import pt2tt, compare_tensors
from test.common import MyDict

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, ATTENTION, NORM_LAYERS

# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #
NORM_LAYERS.register_module("LN_tt", module=op.LayerNorm, force=True)

@pytest.fixture(scope="session")
def device():
    """Open a TT device once per test session and return it."""
    device_box.open()
    dev = device_box.get()
    yield dev
    device_box.close()

@pytest.fixture
def random_inputs(device):
    """Generate deterministic inputs and their TTNN copies."""
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

# --------------------------------------------------------------------------- #
#  Test                                                                        #
# --------------------------------------------------------------------------- #

def test_latent_decoder_equivalence(random_inputs):
    ref_input, input = random_inputs

    # Layer config
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
    
    # Initialise both models
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
    out = model(**input)
    
    # Assertions
    assert ref_out.shape == out.shape, "Output shapes do not match"
    assert compare_tensors(ref_out, out)[0], "Tensor values differ beyond tolerance"

