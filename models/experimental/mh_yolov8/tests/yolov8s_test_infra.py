# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.bos_model.mh_yolov8.tt.ttnn_yolov8s import TtYolov8sModel
from models.bos_model.mh_yolov8.tt.tt_yolov8s_utils import custom_preprocessor
from models.common.utility_functions import divup, is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


def load_torch_model():
    torch_model = YOLO("yolov8s.pt")
    torch_model = torch_model.model
    torch_model.eval()
    return torch_model


def load_ttnn_model(device, torch_model, inp_h, inp_w):
    state_dict = torch_model.state_dict()
    parameters = custom_preprocessor(device, state_dict, inp_h=inp_h, inp_w=inp_w)
    ttnn_model = TtYolov8sModel(device=device, parameters=parameters, res=(inp_h, inp_w))
    return ttnn_model


class Yolov8TestInfra:
    def __init__(
        self,
        device,
        batch_size,
        input_size=(320, 320),
        model_location_generator=None,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.pcc_passed = False
        self.pcc_message = "Did you forget to call validate()?"
        self.device = device
        self.batch_size = batch_size
        self.model_location_generator = model_location_generator
        torch_model = load_torch_model()
        inp_h, inp_w = input_size
        input_shape = (batch_size, inp_h, inp_w, 3)  # NHWC
        self.ttnn_yolov8_model = load_ttnn_model(device=self.device, torch_model=torch_model, inp_h=inp_h, inp_w=inp_w)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        # self.tt_input_tensor = ttnn.from_torch(torch_input_tensor.reshape([1, 1, -1, 3]), ttnn.bfloat16)  # dummy
        self.torch_input_tensor = torch_input_tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
        self.torch_output_tensor = torch_model(self.torch_input_tensor)[0]

    def run(self):
        # input_tensor = ttnn.to_device(self.input_tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.output_tensors = self.ttnn_yolov8_model(self.input_tensor)

    def setup_l1_sharded_input(self, device, torch_input_tensor=None, min_channels=16, num_cores=20):
        if num_cores == 20:
            core_grid = ttnn.CoreGrid(x=5, y=4)
        else:
            core_grid = ttnn.CoreGrid(x=8, y=8)
        torch_input_tensor = self.torch_input_tensor if torch_input_tensor is None else torch_input_tensor

        torch_input_tensor = torch_input_tensor.permute((0, 2, 3, 1))  # NCHW -> NHWC
        n, h, w, c = torch_input_tensor.shape
        if c < min_channels:
            channel_padding_needed = min_channels - c
            torch_input_tensor = torch.nn.functional.pad(
                torch_input_tensor, (0, channel_padding_needed, 0, 0, 0, 0), value=0.0
            )
            c = min_channels
        torch_input_tensor = torch_input_tensor.reshape(1, 1, n * h * w, c)
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, h, w, c],
            core_grid,
            ttnn.ShardStrategy.HEIGHT,
        )
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, torch_input_tensor=None, min_channels=16, num_cores=20):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(
            device, torch_input_tensor=torch_input_tensor, min_channels=min_channels, num_cores=num_cores
        )
        dram_grid_size = device.dram_grid_size()
        dram_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            ),
            [
                divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], (dram_grid_size.x * dram_grid_size.y)),
                tt_inputs_host.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config_DRAM = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
        )

        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def validate(self, output_tensor=None):
        output_tensor = self.output_tensors[0] if output_tensor is None else output_tensor
        output_tensor = ttnn.to_torch(output_tensor)

        valid_pcc = 0.978
        self.pcc_passed, self.pcc_message = assert_with_pcc(self.torch_output_tensor, output_tensor, pcc=valid_pcc)

        logger.info(f"Yolov8s batch_size={self.batch_size}, PCC={self.pcc_message}")

    def dealloc_output(self):
        ttnn.deallocate(self.output_tensors[0])
        _ = [ttnn.deallocate(x) for x in self.output_tensors[1]]


def create_test_infra(
    device,
    batch_size,
    input_size,
):
    return Yolov8TestInfra(
        device,
        batch_size,
        input_size,
    )
