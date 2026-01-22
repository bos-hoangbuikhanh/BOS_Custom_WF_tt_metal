import math

import torch.nn as nn

import ttnn
from models.bos_model.yolov8s.modules.YoloV8FPN import BOS_TTNN_Concat
from models.bos_model.yolov8s.modules.YoloV8Nets import BOS_TTNN_Conv
from models.bos_model.yolov8s.utilities.utility_functions import _nearest_32


class BOS_TTNN_Detect(nn.Module):
    def __init__(self, image_shapes, channels, num_classes, base_address, device, layer_configs={}, pcc_check=None):
        super().__init__()

        self.device = device
        self.image_shapes = image_shapes
        self.channels = channels
        self.num_classes = num_classes
        self.base_address = base_address
        self.pcc_check = pcc_check

        self.reg_max = 16
        c2, c3 = max((16, channels[0] // 4, self.reg_max * 4)), max(channels[0], min(self.num_classes, 100))

        self.cv2_first = nn.ModuleList(
            nn.Sequential(
                BOS_TTNN_Conv(
                    self.image_shapes[i],
                    in_channels,
                    c2,
                    self.base_address + f"cv2.{i}.0.",
                    self.device,
                    3,
                    1,
                    1,
                    return_input=True,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
            )
            for i, in_channels in enumerate(self.channels)
        )
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                BOS_TTNN_Conv(
                    self.image_shapes[i],
                    c2,
                    c2,
                    self.base_address + f"cv2.{i}.1.",
                    self.device,
                    3,
                    1,
                    1,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Conv(
                    self.image_shapes[i],
                    c2,
                    4 * self.reg_max,
                    self.base_address + f"cv2.{i}.2.",
                    self.device,
                    1,
                    1,
                    0,
                    batchnorm=False,
                    activation=False,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
            )
            for i, in_channels in enumerate(self.channels)
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(
                BOS_TTNN_Conv(
                    self.image_shapes[i],
                    in_channels,
                    c3,
                    self.base_address + f"cv3.{i}.0.",
                    self.device,
                    3,
                    1,
                    1,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Conv(
                    self.image_shapes[i],
                    c3,
                    c3,
                    self.base_address + f"cv3.{i}.1.",
                    self.device,
                    3,
                    1,
                    1,
                    layer_configs=layer_configs,
                    # pcc_check=0.995,
                ),
                BOS_TTNN_Conv(
                    self.image_shapes[i],
                    c3,
                    self.num_classes,
                    self.base_address + f"cv3.{i}.2.",
                    self.device,
                    1,
                    1,
                    0,
                    layer_configs=layer_configs,
                    batchnorm=False,
                    activation=False,
                    # pcc_check=0.995,
                ),
            )
            for i, in_channels in enumerate(self.channels)
        )

        self.concat = nn.ModuleList(
            BOS_TTNN_Concat(-1, self.base_address + f"concat.{i}", self.device)
            for i, in_channels in enumerate(self.channels)
        )

    def forward(self, pyramids):
        input_pyramids = []
        output_first = []
        for i in range(len(self.channels)):
            x, output = self.cv2_first[i](pyramids[i])
            input_pyramids.append(x)
            output_first.append(output)

        cv_out = []
        for i in range(len(self.channels)):
            # print(output_first[i].memory_config())
            # print(input_pyramids[i].memory_config())
            cv2_out = self.cv2[i](output_first[i])
            cv3_out = self.cv3[i](input_pyramids[i])

            cv2_out = ttnn.sharded_to_interleaved(cv2_out, ttnn.L1_MEMORY_CONFIG)
            if cv3_out.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
                # cv3_out = ttnn.to_layout(cv2_out, ttnn.ROW_MAJOR_LAYOUT)
                shard_width = _nearest_32(cv3_out.shape[-1])
                nhcores = math.ceil(cv3_out.shape[-2] / 32)
                shard_height = 32 if nhcores <= 20 else 64
                core_range_set = {
                    2: [[(0, 0), (1, 0)]],
                    4: [[(0, 0), (3, 0)]],
                    8: [[(0, 0), (4, 0)], [(0, 1), (2, 1)]],
                    13: [[(0, 0), (4, 1)], [(0, 2), (2, 2)]],
                    16: [[(0, 0), (4, 2)], [(0, 3), (0, 3)]],
                    17: [[(0, 0), (4, 2)], [(0, 3), (1, 3)]],
                }[nhcores]
                shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(cores[0], cores[1]) for cores in core_range_set})
                shard_spec = ttnn.ShardSpec(shard_grid, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR)
                reshard_memory_config = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
                )
                cv3_out = ttnn.to_memory_config(cv3_out, reshard_memory_config)
                cv3_out = ttnn.to_layout(cv3_out, ttnn.TILE_LAYOUT)
            cv3_out = ttnn.sharded_to_interleaved(cv3_out, ttnn.L1_MEMORY_CONFIG)

            # if cv2_out.shape[-2] % 32 != 0:
            #     cv2_out = ttnn.to_layout(cv2_out, ttnn.ROW_MAJOR_LAYOUT)
            #     cv2_out = ttnn.pad(cv2_out, ((0, 0), (0, 0), (0, _nearest_32(cv2_out.shape[-2]) - cv2_out.shape[-2]), (0, 0)), 0)
            #     cv2_out = ttnn.to_layout(cv2_out, ttnn.TILE_LAYOUT)
            # if cv3_out.shape[-2] % 32 != 0:
            #     cv3_out = ttnn.to_layout(cv3_out, ttnn.ROW_MAJOR_LAYOUT)
            #     cv3_out = ttnn.pad(
            #         cv3_out, ((0, 0), (0, 0), (0, _nearest_32(cv3_out.shape[-2]) - cv3_out.shape[-2]), (0, 0)), 0
            #     )
            #     cv3_out = ttnn.to_layout(cv3_out, ttnn.TILE_LAYOUT)
            cv_out.append(self.concat[i](cv2_out, cv3_out))

        return cv_out
