import math
import os

import pytest
import torch

import ttnn

os.makedirs("pts", exist_ok=True)

from test.common import MyDict
from test.configs.op_configs import memory_config, program_config
from test.SSR.reference.tokenlearner import TokenLearnerV11_Torch
from test.SSR.tt.tokenlearner import TokenLearnerV11
from test.utils import compare_tensors, pt2tt, tt2pt

# Third‑party Libraries
from bos_metal import device_box, op


@pytest.fixture(scope="session")
def device():
    """Open a TT device once per test session and return it."""
    dev = device_box.open()
    yield dev
    device_box.close()


def make_random_inputs(device):
    """Generate deterministic inputs and their TTNN copies."""
    torch.manual_seed(0)

    input_torch = torch.randn(1, 10000, 512)
    input_tt = pt2tt(input_torch, device=device)

    return input_torch, input_tt


def make_program_memory_config():
    return program_config["SSRHead"]["tokenlearner"], memory_config["SSRHead"]["tokenlearner"]


@pytest.fixture
def program_memory_config():
    return make_program_memory_config()


# --------------------------------------------------------------------------- #
#  Test                                                                        #
# --------------------------------------------------------------------------- #


def test_tokenlearner_layer_equivalence(device, program_memory_config):
    """
    Verify that SELayerV1 (PyTorch) and SELayerV2 (TTNN) produce numerically
    equivalent outputs for the same inputs.
    """
    ref_x, x = make_random_inputs(device)
    program_config, memory_config = program_memory_config

    # Initialise both models
    # ttnn.device.EnablePersistentKernelCache()
    tokenlearner_torch = TokenLearnerV11_Torch(16, 512)
    tokenlearner_torch.eval()
    state_dict = tokenlearner_torch.state_dict()
    tokenlearner_ttnn = TokenLearnerV11(16, 512, device=device_box.get())
    tokenlearner_ttnn.load_state_dict(state_dict)
    ref_out, ref_selected = tokenlearner_torch(ref_x)

    out = tokenlearner_ttnn(x, memory_config=memory_config, program_config=program_config)

    # TOTAL timer for this pass

    out = tt2pt(out)
    # Assertions

    print(ref_out)
    print(out)
    assert ref_out.shape == out.shape, "Output shapes do not match"
    assert compare_tensors(ref_out, out, message="Output 1"), "Tensor values differ beyond tolerance"


if __name__ == "__main__":
    print("Code test for verify layer by layer:")
    device_box.open()
    dev = device_box.get()
    program_memory_config = make_program_memory_config()
    test_tokenlearner_layer_equivalence(dev, program_memory_config)
