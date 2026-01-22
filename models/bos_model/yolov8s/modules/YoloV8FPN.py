import collections.abc

import torch
import torch.nn as nn

import ttnn
from models.bos_model.yolov8s.utilities.utility_functions import (
    _nearest_32,
    comp_pcc,
    divide,
    dprint,
    torch_pcc,
    tt_to_torch_tensor,
)


class BOS_TTNN_Upsample(nn.Module):
    def __init__(self, output_shape, channels, base_address, device, pcc_check=None):
        super().__init__()

        self.device = device
        self.output_shape = output_shape
        self.input_shape = divide(output_shape, 2)
        self.channels = channels
        self.base_address = base_address
        self.pcc_check = pcc_check

    def Golden_Upsample2d(self, input_tensor, scale_factor=None, size=None, mode="nearest", align_corners=False):
        return nn.functional.interpolate(
            input_tensor,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners if mode in ["linear", "bilinear", "bicubic", "trilinear"] else None,
        )

    def forward(self, x, scale_factor=[2, 2, 1], mode="nearest"):
        if x.shape != self.input_shape[0] * self.input_shape[1]:
            x = x[:, :, : self.input_shape[0] * self.input_shape[1], :]
        x = ttnn.reshape(x, (1, self.input_shape[0], self.input_shape[1], self.channels))
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        if self.pcc_check is not None:
            torch_input = tt_to_torch_tensor(x)
            torch_input = (
                torch_input.permute(0, 3, 1, 2)
                .reshape(1, self.channels, self.input_shape[0], self.input_shape[1])
                .to(torch.float32)
            )

        output = ttnn.upsample(x, scale_factor=[2, 2], mode=mode)

        if self.pcc_check is not None:
            torch_output = tt_to_torch_tensor(output).permute(0, 3, 1, 2)
            golden_output = self.Golden_Upsample2d(torch_input, (2, 2))
            pcc = torch_pcc(golden_output, torch_output)
            print(
                ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",
                self.base_address,
                "Upsample PCC =",
                pcc,
            )
            assert pcc >= self.pcc_check

        output = ttnn.reshape(output, (1, 1, self.output_shape[0] * self.output_shape[1], self.channels))

        return output


class BOS_TTNN_Concat(nn.Module):
    def __init__(
        self, dim, base_address, device, input_a_shape=None, input_b_shape=None, layer_configs={}, pcc_check=None
    ):
        super().__init__()

        self.device = device
        self.dim = dim
        self.base_address = base_address + ".concat"
        self.pcc_check, self.input_a_shape, self.input_b_shape = pcc_check, input_a_shape, input_b_shape

        try:
            self.output_mem_config = layer_configs[self.base_address]
        except:
            self.output_mem_config = None

    def Golden_Concat(self, input_tensors, dim):
        return torch.cat(input_tensors, dim)

    def forward(self, x, y, pad_if_optimal=False, out_shard_if_optimal=False):
        if (self.pcc_check is not None) and (self.input_a_shape is not None) and (self.input_b_shape is not None):
            torch_input_a = tt_to_torch_tensor(x)
            torch_input_a = torch_input_a.permute(0, 3, 1, 2).reshape(self.input_a_shape).to(torch.float32)
            torch_input_b = tt_to_torch_tensor(x)
            torch_input_b = torch_input_b.permute(0, 3, 1, 2).reshape(self.input_b_shape).to(torch.float32)

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if pad_if_optimal and (x.shape[-2] % 32 != 0):
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, _nearest_32(x.shape[-2]) - x.shape[-2]), (0, 0)], 0)
        y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        if pad_if_optimal and (y.shape[-2] % 32 != 0):
            y = ttnn.pad(y, [(0, 0), (0, 0), (0, _nearest_32(y.shape[-2]) - y.shape[-2]), (0, 0)], 0)

        # output = ttnn.concat([x, y], memory_config=self.output_mem_config, dim=self.dim)
        output = ttnn.concat([x, y], dim=self.dim)
        if self.output_mem_config is not None:
            output = ttnn.to_memory_config(output, self.output_mem_config)

        if (self.pcc_check is not None) and (self.input_a_shape is not None) and (self.input_b_shape is not None):
            torch_output = tt_to_torch_tensor(output).to(torch.float32)
            golden_output = self.Golden_Concat([torch_input_a, torch_input_b], self.dim)
            if golden_output.shape[-1] != torch_output.shape[-1]:
                golden_output = nn.functional.pad(golden_output, (0, torch_output.shape[-1] - golden_output.shape[-1]))
            print(golden_output.shape, torch_output.shape)
            pcc = torch_pcc(golden_output, torch_output)
            print(
                ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",
                self.base_address,
                "Concat PCC =",
                pcc,
            )
            assert pcc >= self.pcc_check

        return output
