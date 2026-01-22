# tests/test_se_layer_equivalence.py
import pytest
import torch
import ttnn

from bos_metal import op, device_box, engine
from test.utils import pt2tt, compare_tensors
from test.common import ReLU, Sigmoid, MyDict

# --------------------------------------------------------------------------- #
#  The two implementations under test                                          #
# --------------------------------------------------------------------------- #

from test.SSR.reference.squeeze_excitation import SELayer as SELayerV1
from test.SSR.tt.squeeze_excitation import SELayer as SELayerV2

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
    seq_len = 10_000
    num_channels = 256

    ref_x = torch.randn(batch_size, seq_len, num_channels)
    x = pt2tt(ref_x,
              memory_config=ttnn.L1_MEMORY_CONFIG,
              device=device)

    ref_x_se = torch.randn(1, 1, num_channels)
    x_se = pt2tt(ref_x_se,
                 memory_config=ttnn.L1_MEMORY_CONFIG,
                 device=device)

    return num_channels, ref_x, ref_x_se, x, x_se

# --------------------------------------------------------------------------- #
#  Test                                                                        #
# --------------------------------------------------------------------------- #

def test_se_layer_equivalence(device):
    """
    Verify that SELayerV1 (PyTorch) and SELayerV2 (TTNN) produce numerically
    equivalent outputs for the same inputs.
    """
    num_channels, ref_x, ref_x_se, x, x_se = random_inputs(device)

    # Initialise both models
    ref_model = SELayerV1(num_channels)
    model = SELayerV2(num_channels)

    # Load processed state dict into TTNN model
    state_dict = engine.ModelProcessor(ref_model)\
                       .process_state_dict(x=ref_x, x_se=ref_x_se)
    model.load_state_dict(state_dict, strict=False)

    # Forward passes
    ref_out = ref_model(ref_x, ref_x_se)
    out = model(x, x_se)
    
    # Assertions
    assert ref_out.shape == out.shape, "Output shapes do not match"
    assert compare_tensors(ref_out, out), "Tensor values differ beyond tolerance"
