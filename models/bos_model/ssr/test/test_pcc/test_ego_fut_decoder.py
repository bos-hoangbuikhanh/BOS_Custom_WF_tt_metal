# tests/test_se_layer_equivalence.py
import pytest
import torch
import ttnn

from bos_metal import op, device_box, engine
from test.utils import pt2tt, compare_tensors
from test.common import ReLU, Sigmoid, MyDict

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

@pytest.fixture
def random_inputs(device):
    """Generate deterministic inputs and their TTNN copies."""
    torch.manual_seed(0)

    batch_size = 1
    num_channels = 256
    num_reg_fcs = 2

    ref_x = torch.randn(batch_size, 18, num_channels)
    x = pt2tt(ref_x,
              memory_config=ttnn.L1_MEMORY_CONFIG,
              device=device)

    return ref_x, x, num_reg_fcs

# --------------------------------------------------------------------------- #
#  Test                                                                        #
# --------------------------------------------------------------------------- #

def create_model(num_reg_fcs, Linear, ReLU, embed_dims=256):
    ego_fut_decoder = []
    for _ in range(num_reg_fcs):
        # TODO: Fuse Linear and ReLU
        ego_fut_decoder.append(Linear(embed_dims, embed_dims))
        ego_fut_decoder.append(ReLU())
    ego_fut_decoder.append(Linear(embed_dims, 2))
    
    return torch.nn.Sequential(*ego_fut_decoder) 
    

def test_ego_fut_decoder_equivalence(random_inputs):
    ref_input, input, num_reg_fcs = random_inputs

    # Initialise both models
    ref_model = create_model(num_reg_fcs, torch.nn.Linear, torch.nn.ReLU)
    model = create_model(num_reg_fcs, op.Linear, ReLU)

    # Load processed state dict into TTNN model
    state_dict = engine.ModelProcessor(ref_model)\
                       .process_state_dict(ref_input)
    model.load_state_dict(state_dict, strict=False)
    
    # Forward passes
    ref_out = ref_model(ref_input)
    out = model(input)
    
    # Assertions
    assert ref_out.shape == out.shape, "Output shapes do not match"
    assert compare_tensors(ref_out, out), "Tensor values differ beyond tolerance"
