import torch
from ttnn.model_preprocessing import preprocess_model_parameters
import pytest
import ttnn


def test_hanging_silce(device):
    inp_shape = (2048 * 32, 2)
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    print("Creating input tensor")
    inp = torch.rand(inp_shape)
    print("Converting to TTNN tensor")
    inp_tt = ttnn.from_torch(inp, memory_config=memory_config, device=device)
    print("Running Getitem")
    output = inp_tt[:, 1:2]
    print("Converting to torch")
    output = ttnn.to_torch(output)
    print(output)
