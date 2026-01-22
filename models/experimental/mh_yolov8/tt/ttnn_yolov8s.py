# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
from models.bos_model.mh_yolov8.tt.tt_yolov8s_utils import ttnn_decode_bboxes
from models.bos_model.mh_yolov8.yolo_common.yolo_utils import determine_num_cores, get_core_grid_from_num_cores
import json
import torch

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

with open("models/bos_model/mh_yolov8/tt/configs.json", "r") as file:
    configs = json.load(file)

conv_config = configs["conv_config"]
sppf_configs = configs["sppf_configs"]
c2f_configs = configs["c2f_configs"]
detect_config = configs["detect_config"]

save_pt = False
regen_input = False
N = 1


def sharded_concat_sppf(input_tensors, num_cores=20, dim=3):  # expected input tensors to be in fp16, RM, same (h*w)
    if use_signpost:
        signpost(header="sharded_concat_sppf")

    shard_height = (input_tensors[0].shape[2] + num_cores - 1) // num_cores

    input_sharded_memory_configs = []

    for i in range(len(input_tensors)):
        input_sharded_memory_config = ttnn.create_sharded_memory_config(
            (shard_height, input_tensors[i].shape[-1]),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_sharded_memory_configs.append(input_sharded_memory_config)

    sharded_inputs = [
        ttnn.to_memory_config(tensor, config) for tensor, config in zip(input_tensors, input_sharded_memory_configs)
    ]

    total_width = sum(tensor.shape[-1] for tensor in input_tensors)
    out_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, total_width),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    output = ttnn.concat(sharded_inputs, dim, memory_config=out_sharded_memory_config)
    output = ttnn.sharded_to_interleaved(output, memory_config=ttnn.L1_MEMORY_CONFIG)

    return output


def sharded_concat(
    input_tensors, num_cores=20, dim=3, skip_s2i=False
):  # expected input tensors to be in fp16, RM, same (h*w)
    if use_signpost:
        signpost(header="sharded_concat")

    shard_height = (input_tensors[0].shape[2] + num_cores - 1) // num_cores

    input_sharded_memory_configs = []

    for i in range(len(input_tensors)):
        input_sharded_memory_config = ttnn.create_sharded_memory_config(
            (shard_height, input_tensors[i].shape[-1]),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_sharded_memory_configs.append(input_sharded_memory_config)

    sharded_inputs = [
        ttnn.to_memory_config(tensor, config) for tensor, config in zip(input_tensors, input_sharded_memory_configs)
    ]

    total_width = sum(tensor.shape[-1] for tensor in input_tensors)
    out_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, total_width),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    output = ttnn.concat(sharded_inputs, dim, memory_config=out_sharded_memory_config)
    if not skip_s2i:
        output = ttnn.sharded_to_interleaved(output, memory_config=ttnn.L1_MEMORY_CONFIG)

    return output


class TtConv:
    def __init__(
        self,
        device,
        parameters,
        path,
        input_params,
        groups=1,
        dilation=1,
        act_block_h=False,
        block_shard=None,
        bfloat8=True,
        change_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        is_fused=True,
        is_detect_cv2=False,
        width_shard=False,
        act_blocks=32,
        enable_act_double_buffer=True,
        enable_split_reader=False,
        reshard_if_not_optimal=False,
        batch_size=1,
    ):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.input_params = input_params
        self.groups = groups
        self.dilation = dilation
        self.act_block_h = act_block_h
        self.block_shard = block_shard
        self.bfloat8 = bfloat8
        self.change_shard = change_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout
        self.is_fused = is_fused
        self.is_detect_cv2 = is_detect_cv2
        self.width_shard = width_shard
        self.act_blocks = act_blocks
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_split_reader = enable_split_reader
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.batch_size = batch_size

        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters[path]

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            activation="" if self.is_detect_cv2 else "silu",
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            act_block_w_div=1,
            transpose_shards=False,
            deallocate_activation=False,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_split_reader=self.enable_split_reader,
            enable_subblock_padding=False,
            output_layout=self.output_layout,
            reallocate_halo_output=False,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
        )

        if self.deallocate_activation:
            conv_config.deallocate_activation = self.deallocate_activation

        if self.change_shard:
            conv_config.shard_layout = None

        if self.act_block_h:
            conv_config.act_block_h_override = self.act_blocks

        if self.block_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        if self.width_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

        if self.bfloat8:
            conv_config.weights_dtype = ttnn.bfloat8_b
            conv_config.dtype = ttnn.bfloat8_b

        return conv_config

    def _initialize_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        if x.shape[1] != 1:
            input_height = x.shape[1]
            input_width = x.shape[2]
        else:
            input_height = int(math.sqrt(x.shape[2]) // self.batch_size)
            input_width = int(math.sqrt(x.shape[2]) // self.batch_size)

        [x, [out_height, out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            in_channels=self.input_params[4],
            out_channels=self.input_params[3],
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=(self.input_params[0], self.input_params[0]),
            stride=(self.input_params[1], self.input_params[1]),
            padding=(self.input_params[2], self.input_params[2]),
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            memory_config=None,
            return_weights_and_bias=True,
            return_output_dim=True,
        )

        if self.is_detect_cv2:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            return x, out_height, out_width

        return x, out_height, out_width


class TtBottleneck:
    def __init__(
        self,
        device,
        parameters,
        path,
        shortcut,
        change_shard,
        input_params,
        act_block_h=False,
        block_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        tilize=False,
    ):
        self.device = device
        self.path = path
        self.tilize = tilize
        self.shortcut = shortcut
        self.block_shard = block_shard
        self.cv1 = TtConv(
            device,
            parameters,
            f"{self.path}.cv1",
            input_params,
            change_shard=change_shard,
            block_shard=self.block_shard,
            deallocate_activation=deallocate_activation,
            output_layout=output_layout,
        )
        self.cv2 = TtConv(
            device,
            parameters,
            f"{self.path}.cv2",
            input_params,
            act_block_h=act_block_h,
            change_shard=change_shard,
            block_shard=self.block_shard,
            deallocate_activation=deallocate_activation,
        )

    def __call__(self, x):
        if use_signpost:
            signpost(header="BottleNeck")

        cv1, out_h, out_w = self.cv1(x)
        if save_pt:
            c2f_id = self.path.split(".")[-3]
            btn_id = self.path.split(".")[-1]
            torch_cv1 = torch.Tensor(ttnn.to_torch(ttnn.from_device(cv1)))
            torch_cv1 = torch_cv1.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_cv1), f"ttnn_c2f_{c2f_id}.m{btn_id}.cv1.pt")
        if regen_input:
            c2f_id = self.path.split(".")[-3]
            btn_id = self.path.split(".")[-1]
            cv1_dtype = cv1.dtype
            cv1_layout = cv1.layout
            cv1_memory_config = cv1.memory_config()
            torch_cv1 = torch.load(f"torch_c2f_{c2f_id}.m{btn_id}.cv1.pt").permute(0, 2, 3, 1).reshape(tuple(cv1.shape))
            ttnn.deallocate(cv1)
            cv1 = ttnn.from_torch(
                torch_cv1, dtype=cv1_dtype, layout=cv1_layout, memory_config=cv1_memory_config, device=self.device
            )
        cv2, out_h, out_w = self.cv2(cv1)  # pass cv1
        if save_pt:
            c2f_id = self.path.split(".")[-3]
            btn_id = self.path.split(".")[-1]
            torch_cv2 = torch.Tensor(ttnn.to_torch(ttnn.from_device(cv2)))
            torch_cv2 = torch_cv2.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_cv2), f"ttnn_c2f_{c2f_id}.m{btn_id}.cv2.pt")
        if regen_input:
            c2f_id = self.path.split(".")[-3]
            btn_id = self.path.split(".")[-1]
            cv2_dtype = cv2.dtype
            cv2_layout = cv2.layout
            cv2_memory_config = cv2.memory_config()
            torch_cv2 = torch.load(f"torch_c2f_{c2f_id}.m{btn_id}.cv2.pt").permute(0, 2, 3, 1).reshape(tuple(cv2.shape))
            ttnn.deallocate(cv2)
            cv2 = ttnn.from_torch(
                torch_cv2, dtype=cv2_dtype, layout=cv2_layout, memory_config=cv2_memory_config, device=self.device
            )
        ttnn.deallocate(cv1)

        if self.tilize:
            x = ttnn.to_layout(
                x, ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
            )

        # return ttnn.add(x, cv2, memory_config=ttnn.L1_MEMORY_CONFIG) if self.shortcut else cv2
        return (ttnn.add(x, cv2) if self.shortcut else cv2, out_h, out_w)


class TtC2f:
    def __init__(
        self,
        device,
        parameters,
        path,
        n=1,
        shortcut=False,
        change_shard=None,
        input_params=None,
        act_block_h=False,
        bfloat8=True,
        block_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    ):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.n = n
        self.shortcut = shortcut
        self.change_shard = change_shard
        self.input_params = input_params
        self.act_block_h = act_block_h
        self.bfloat8 = bfloat8
        self.block_shard = block_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout

        self.cv1_a = TtConv(
            device,
            self.parameters,
            f"{self.path}.cv1_a",
            input_params=self.input_params[3],
            bfloat8=self.bfloat8,
            change_shard=self.change_shard,
            deallocate_activation=self.deallocate_activation,
            output_layout=self.output_layout,
        )

        self.cv1_b = TtConv(
            device,
            self.parameters,
            f"{self.path}.cv1_b",
            input_params=self.input_params[4],
            bfloat8=self.bfloat8,
            change_shard=self.change_shard,
            deallocate_activation=self.deallocate_activation,
            output_layout=self.output_layout,
        )

        self.cv2 = TtConv(
            self.device,
            self.parameters,
            f"{self.path}.cv2",
            input_params=self.input_params[1],
            bfloat8=self.bfloat8,
            block_shard=self.block_shard,
            change_shard=self.change_shard,
            deallocate_activation=self.deallocate_activation,
        )

        self.bottleneck_modules = []
        for i in range(self.n):
            self.tilize = i == 0
            self.bottleneck_modules.append(
                TtBottleneck(
                    self.device,
                    self.parameters,
                    f"{self.path}.m.{i}",
                    self.shortcut,
                    self.change_shard,
                    input_params=self.input_params[2],
                    act_block_h=self.act_block_h,
                    block_shard=self.block_shard,
                    deallocate_activation=self.deallocate_activation,
                    tilize=self.tilize,
                )
            )

    def __call__(self, x, reshard_bottleneck_input=False):
        if use_signpost:
            signpost(header="C2F")

        cv1_a, out_h_a, out_w_a = self.cv1_a(x)
        cv1_b, out_h_b, out_w_b = self.cv1_b(x)
        if save_pt:
            c2f_id = self.path.split(".")[-1]
            torch_cv1_a = torch.Tensor(ttnn.to_torch(ttnn.from_device(cv1_a)))
            torch_cv1_a = torch_cv1_a.reshape([N, out_h_a, out_w_a, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_cv1_a), f"ttnn_c2f_{c2f_id}.cv1_a.pt")
            torch_cv1_b = torch.Tensor(ttnn.to_torch(ttnn.from_device(cv1_b)))
            torch_cv1_b = torch_cv1_b.reshape([N, out_h_b, out_w_b, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_cv1_b), f"ttnn_c2f_{c2f_id}.cv1_b.pt")
        if regen_input:
            c2f_id = self.path.split(".")[-1]
            cv1_a_dtype = cv1_a.dtype
            cv1_a_layout = cv1_a.layout
            cv1_a_memory_config = cv1_a.memory_config()
            torch_cv1_a = torch.load(f"torch_c2f_{c2f_id}.cv1_a.pt").permute(0, 2, 3, 1).reshape(tuple(cv1_a.shape))
            ttnn.deallocate(cv1_a)
            cv1_a = ttnn.from_torch(
                torch_cv1_a,
                dtype=cv1_a_dtype,
                layout=cv1_a_layout,
                memory_config=cv1_a_memory_config,
                device=self.device,
            )
            cv1_b_dtype = cv1_b.dtype
            cv1_b_layout = cv1_b.layout
            cv1_b_memory_config = cv1_b.memory_config()
            torch_cv1_b = torch.load(f"torch_c2f_{c2f_id}.cv1_b.pt").permute(0, 2, 3, 1).reshape(tuple(cv1_b.shape))
            ttnn.deallocate(cv1_b)
            cv1_b = ttnn.from_torch(
                torch_cv1_b,
                dtype=cv1_b_dtype,
                layout=cv1_b_layout,
                memory_config=cv1_b_memory_config,
                device=self.device,
            )

        if reshard_bottleneck_input:
            cv1_a = ttnn.to_memory_config(cv1_a, ttnn.L1_MEMORY_CONFIG)
            cv1_b = ttnn.to_memory_config(cv1_b, ttnn.L1_MEMORY_CONFIG)

        y = [cv1_a, cv1_b]

        # if reshard_bottleneck_input:
        #     bottlenech_input = ttnn.to_memory_config(y[-1], ttnn.L1_MEMORY_CONFIG)
        # else:
        #     bottlenech_input = y[-1]

        for i in range(self.n):
            z, z_h, z_w = self.bottleneck_modules[i](y[-1])
            if (not self.shortcut) and reshard_bottleneck_input:
                z = ttnn.to_memory_config(z, ttnn.L1_MEMORY_CONFIG)
            if save_pt:
                c2f_id = self.path.split(".")[-1]
                torch_btn = torch.Tensor(ttnn.to_torch(ttnn.from_device(z)))
                torch_btn = torch_btn.reshape([N, z_h, z_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
                torch.save(torch.Tensor(torch_btn), f"ttnn_c2f_{c2f_id}.m{i}.add.pt")
            if regen_input:
                c2f_id = self.path.split(".")[-1]
                z_dtype = z.dtype
                z_layout = z.layout
                z_memory_config = z.memory_config()
                torch_z = torch.load(f"torch_c2f_{c2f_id}.m{i}.add.pt").permute(0, 2, 3, 1).reshape(tuple(z.shape))
                ttnn.deallocate(z)
                z = ttnn.from_torch(
                    torch_z, dtype=z_dtype, layout=z_layout, memory_config=z_memory_config, device=self.device
                )
            y.append(z)

        total_width = sum(tensor.shape[-1] for tensor in y)

        if not reshard_bottleneck_input:
            # if y[0].is_sharded():
            input_sharded_memory_config = y[0].memory_config()
            input_sharded_spec = input_sharded_memory_config.shard_spec
            shard_height = input_sharded_spec.shape[0]
            input_sharded_memory_layout = str(input_sharded_memory_config.memory_layout)
            if "HEIGHT" in input_sharded_memory_layout:
                out_sharded_memory_config_strategy = ttnn.ShardStrategy.HEIGHT
            elif "BLOCK" in input_sharded_memory_layout:
                out_sharded_memory_config_strategy = ttnn.ShardStrategy.BLOCK
            elif "WIDTH" in input_sharded_memory_layout:
                out_sharded_memory_config_strategy = ttnn.ShardStrategy.WIDTH

            out_sharded_memory_config = ttnn.create_sharded_memory_config(
                (shard_height, total_width),
                core_grid=input_sharded_spec.grid,
                strategy=out_sharded_memory_config_strategy,
                use_height_and_width_as_shard_shape=True,
            )
            x = ttnn.concat(y, -1, memory_config=out_sharded_memory_config)
        else:
            x = ttnn.concat(y, -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        if save_pt:
            c2f_id = self.path.split(".")[-1]
            torch_c2f_2_y = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_c2f_2_y = torch_c2f_2_y.reshape([N, out_h_a, out_w_a, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_c2f_2_y), f"ttnn_c2f_{c2f_id}.concat.pt")
        if regen_input:
            c2f_id = self.path.split(".")[-1]
            x_dtype = x.dtype
            x_layout = x.layout
            x_memory_config = x.memory_config()
            torch_x = torch.load(f"torch_c2f_{c2f_id}.concat.pt").permute(0, 2, 3, 1).reshape(tuple(x.shape))
            ttnn.deallocate(x)
            x = ttnn.from_torch(
                torch_x, dtype=x_dtype, layout=x_layout, memory_config=x_memory_config, device=self.device
            )

        for i in range(len(y)):
            ttnn.deallocate(y[i])

        x, out_h, out_w = self.cv2(x)
        if save_pt:
            torch_cv2 = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_cv2 = torch_cv2.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_cv2), f"ttnn_c2f_{c2f_id}.cv2.pt")
        return x, out_h, out_w


class TtSppf:
    def __init__(self, device, parameters, path, input_params, batch_size):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.input_params = input_params
        self.batch_size = batch_size

        self.cv1 = TtConv(
            device, parameters, f"{path}.cv1", input_params=input_params[0], change_shard=True, block_shard=True
        )
        self.cv2 = TtConv(
            device, parameters, f"{path}.cv2", input_params=input_params[1], change_shard=True, block_shard=True
        )

    def __call__(self, x):
        if use_signpost:
            signpost(header="SPPF")
        cv1, out_h, out_w = self.cv1(x)
        cv1 = ttnn.to_memory_config(cv1, ttnn.L1_MEMORY_CONFIG)
        cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT)
        if save_pt:
            sppf_id = self.path.split(".")[-1]
            torch_cv1 = torch.Tensor(ttnn.to_torch(ttnn.from_device(cv1)))
            torch_cv1 = torch_cv1.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_cv1), f"ttnn_sppf_{sppf_id}.cv1.pt")
        if regen_input:
            sppf_id = self.path.split(".")[-1]
            cv1_dtype = cv1.dtype
            cv1_layout = cv1.layout
            cv1_memory_config = cv1.memory_config()
            torch_cv1 = torch.load(f"torch_sppf_{sppf_id}.cv1.pt").permute(0, 2, 3, 1).reshape(tuple(cv1.shape))
            ttnn.deallocate(cv1)
            cv1 = ttnn.from_torch(
                torch_cv1, dtype=cv1_dtype, layout=cv1_layout, memory_config=cv1_memory_config, device=self.device
            )
        y = [cv1]

        for i in range(3):
            output = ttnn.max_pool2d(
                input_tensor=y[-1],
                batch_size=self.batch_size,
                input_h=out_h,
                input_w=out_w,
                channels=y[-1].shape[-1],
                kernel_size=[5, 5],
                stride=[1, 1],
                padding=[2, 2],
                dilation=[1, 1],
            )
            if save_pt:
                sppf_id = self.path.split(".")[-1]
                torch_output = torch.Tensor(ttnn.to_torch(ttnn.from_device(output)))
                torch_output = torch_output.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
                torch.save(torch.Tensor(torch_output), f"ttnn_sppf_{sppf_id}.maxpool{i}.pt")
            if regen_input:
                sppf_id = self.path.split(".")[-1]
                output_dtype = output.dtype
                output_layout = output.layout
                output_memory_config = output.memory_config()
                torch_output = (
                    torch.load(f"torch_sppf_{sppf_id}.maxpool{i}.pt").permute(0, 2, 3, 1).reshape(tuple(output.shape))
                )
                ttnn.deallocate(output)
                output = ttnn.from_torch(
                    torch_output,
                    dtype=output_dtype,
                    layout=output_layout,
                    memory_config=output_memory_config,
                    device=self.device,
                )
            y.append(output)

        x = sharded_concat_sppf(y)
        for i in range(len(y)):
            ttnn.deallocate(y[i])
        if save_pt:
            sppf_id = self.path.split(".")[-1]
            torch_concat = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_concat = torch_concat.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_concat), f"ttnn_sppf_{sppf_id}.concat.pt")
        if regen_input:
            sppf_id = self.path.split(".")[-1]
            concat_dtype = x.dtype
            concat_layout = x.layout
            concat_memory_config = x.memory_config()
            torch_concat = torch.load(f"torch_sppf_{sppf_id}.concat.pt").permute(0, 2, 3, 1).reshape(tuple(x.shape))
            ttnn.deallocate(x)
            x = ttnn.from_torch(
                torch_concat,
                dtype=concat_dtype,
                layout=concat_layout,
                memory_config=concat_memory_config,
                device=self.device,
            )

        x, out_h, out_w = self.cv2(x)
        if save_pt:
            sppf_id = self.path.split(".")[-1]
            torch_cv2 = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_cv2 = torch_cv2.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_cv2), f"ttnn_sppf_{sppf_id}.cv2.pt")
        if regen_input:
            sppf_id = self.path.split(".")[-1]
            cv2_dtype = x.dtype
            cv2_layout = x.layout
            cv2_memory_config = x.memory_config()
            torch_cv2 = torch.load(f"torch_sppf_{sppf_id}.cv2.pt").permute(0, 2, 3, 1).reshape(tuple(x.shape))
            ttnn.deallocate(x)
            x = ttnn.from_torch(
                torch_cv2, dtype=cv2_dtype, layout=cv2_layout, memory_config=cv2_memory_config, device=self.device
            )
        return x, out_h, out_w


class TtDetectCv2:
    def __init__(self, device, parameters, path, input_params, block_shard=False):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.input_params = input_params
        self.conv0 = TtConv(
            device, parameters, f"{path}.0", input_params=input_params[0], bfloat8=True, block_shard=block_shard
        )
        self.conv1 = TtConv(
            device, parameters, f"{path}.1", input_params=input_params[1], bfloat8=True, block_shard=block_shard
        )
        self.conv2 = TtConv(
            device,
            parameters,
            path,
            input_params=input_params[2],
            bfloat8=True,
            is_fused=False,
            change_shard=True,
            block_shard=block_shard,
            is_detect_cv2=True,
        )

    def __call__(self, x):
        if use_signpost:
            signpost(header="DetectCv2")
        x, out_h, out_w = self.conv0(x)
        if save_pt:
            _, _, cv_id, detect_id = self.path.split(".")
            torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_x = torch_x.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_x), f"ttnn_detect_22.{detect_id}.{cv_id}.0.pt")
        x, out_h, out_w = self.conv1(x)
        if save_pt:
            torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_x = torch_x.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_x), f"ttnn_detect_22.{detect_id}.{cv_id}.1.pt")
        x, out_h, out_w = self.conv2(x)
        if save_pt:
            torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_x = torch_x.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_x), f"ttnn_detect_22.{detect_id}.{cv_id}.2.pt")
        return x, out_h, out_w


class TtDFL:
    def __init__(self, device, parameters, path, input_params):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.input_params = input_params
        self.conv = TtConv(device, parameters, path, input_params, bfloat8=True, is_fused=False, change_shard=False)

    def __call__(self, x, c1=16):
        if use_signpost:
            signpost(header="DFL")
        b, _, a = x.shape
        x = ttnn.reshape(x, (b, 4, c1, a), memory_config=ttnn.L1_MEMORY_CONFIG)
        if save_pt:
            torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x))).to(torch.float32)
            torch.save(torch.Tensor(torch_x), "ttnn_detect_22._inference.dfl.reshape.pt")
        # if regen_input:  # TODO if need
        x = ttnn.softmax(x, dim=2)
        if save_pt:
            torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x))).to(torch.float32)
            torch.save(torch.Tensor(torch_x), "ttnn_detect_22._inference.dfl.softmax.pt")
        x = ttnn.permute(x, (0, 1, 3, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        x, h, w = self.conv(x)
        if save_pt:
            torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_x = torch_x.reshape([1, h, w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_x), "ttnn_detect_22._inference.dfl.conv.pt")
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        if save_pt:
            torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_x = torch_x.reshape([1, h, w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_x), "ttnn_detect_22._inference.dfl.conv.sharded_to_interleaved.pt")
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if save_pt:
            torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_x = torch_x.reshape([1, h, w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_x), "ttnn_detect_22._inference.dfl.conv.to_layout.pt")
        # ver1) (reshape -> permute) -> reshape
        x = ttnn.reshape(x, (x.shape[0], 4, int(x.shape[2] / 4), x.shape[3]))
        if save_pt:
            torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x))).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_x), "ttnn_detect_22._inference.dfl.conv.reshape.pt")
        x = ttnn.permute(x, (0, 3, 1, 2))
        if save_pt:
            torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x))).to(torch.float32)
            torch.save(torch.Tensor(torch_x), "ttnn_detect_22._inference.dfl.conv.permute.pt")
        # ver2) (permute -> reshape) -> reshape
        # x = ttnn.permute(x, (0, 3, 1, 2))
        # if save_pt:
        #     torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
        #     torch_x = torch_x.reshape([1, -1, h, w]).to(torch.float32)
        #     torch.save(torch.Tensor(torch_x), "ttnn_detect_22._inference.dfl.conv.permute.pt")
        # x = ttnn.reshape(x, (x.shape[0], x.shape[1], 4, int(x.shape[3] / 4)))
        # if save_pt:
        #     torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x))).to(torch.float32)
        #     torch.save(torch.Tensor(torch_x), "ttnn_detect_22._inference.dfl.conv.reshape.pt")
        x = ttnn.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        if save_pt:
            torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(x))).to(torch.float32)
            torch.save(torch.Tensor(torch_x), "ttnn_detect_22._inference.dfl.out.pt")
        return x


class TtDetect:
    def __init__(self, device, parameters, path, input_params, nc=80, ch=(128, 256, 512)):
        self.device = device
        self.parameters = parameters
        self.path = path
        self.input_params = input_params
        self.nc = nc
        self.ch = ch  # ch are already set in the below input_params
        self.detect_cv2_modules = []
        self.detect_cv3_modules = []

        nl = len(self.ch)
        block_shard = False
        for i in range(nl):
            cv2_params = input_params["cv2_params"][i]["input_params"]
            cv3_params = input_params["cv3_params"][i]["input_params"]
            # if i == nl - 1:
            #     block_shard = True
            self.detect_cv2_modules.append(
                TtDetectCv2(device, parameters, f"{path}.cv2.{i}", input_params=cv2_params, block_shard=block_shard)
            )
            self.detect_cv3_modules.append(
                TtDetectCv2(device, parameters, f"{path}.cv3.{i}", input_params=cv3_params, block_shard=block_shard)
            )

        self.dfl_module = TtDFL(
            device, parameters, f"{path}.dfl", input_params=input_params["dfl_params"]["input_params"]
        )

    def __call__(self, x, nc=80, ch=(), reg_max=16):
        nc = self.nc
        ch = self.ch
        nl = len(ch)
        no = nc + reg_max * 4

        for i in range(nl):
            if use_signpost:
                signpost(header=f"Detect {i+1}/{nl}")
            a, a_h, a_w = self.detect_cv2_modules[i](x[i])
            b, b_h, b_w = self.detect_cv3_modules[i](x[i])
            if use_signpost:
                signpost(header="Detect - concat")
            x[i] = ttnn.concat((a, b), dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
            if save_pt:
                torch_detect_22_a = torch.Tensor(ttnn.to_torch(ttnn.from_device(a)))
                torch_detect_22_a = torch_detect_22_a.reshape([N, a_h, a_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
                torch.save(torch.Tensor(torch_detect_22_a), f"ttnn_detect_22.{i}.cv2.pt")
                torch_detect_22_b = torch.Tensor(ttnn.to_torch(ttnn.from_device(b)))
                torch_detect_22_b = torch_detect_22_b.reshape([N, b_h, b_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
                torch.save(torch.Tensor(torch_detect_22_b), f"ttnn_detect_22.{i}.cv3.pt")
                torch_detect_22_xi = torch.Tensor(ttnn.to_torch(ttnn.from_device(x[i])))
                torch_detect_22_xi = torch_detect_22_xi.reshape([N, a_h, a_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
                torch.save(torch.Tensor(torch_detect_22_xi), f"ttnn_detect_22.{i}.concat.pt")
            if regen_input:
                x_dtype = x[i].dtype
                x_layout = x[i].layout
                x_memory_config = x[i].memory_config()
                torch_x = torch.load(f"torch_detect_22.{i}.concat.pt").permute(0, 2, 3, 1).reshape(tuple(x[i].shape))
                ttnn.deallocate(x[i])
                x[i] = ttnn.from_torch(
                    torch_x, dtype=x_dtype, layout=x_layout, memory_config=x_memory_config, device=self.device
                )

        shape = x[0].shape
        anchors, strides = self.parameters["anchors"], self.parameters["strides"]

        if use_signpost:
            signpost(header="Detect - post-processing")
        xi = []
        for i in x:
            i = ttnn.reshape(i, (shape[0], -1, no), memory_config=ttnn.L1_MEMORY_CONFIG)
            xi.append(i)

        # _inference
        x_cat = ttnn.concat(xi, 1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_cat = ttnn.permute(x_cat, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        if save_pt:
            torch_cat = torch.Tensor(ttnn.to_torch(ttnn.from_device(x_cat))).to(torch.float32)
            torch.save(torch.Tensor(torch_cat), "ttnn_detect_22._inference.concat.pt")
        if regen_input:
            x_cat_dtype = x_cat.dtype
            x_cat_layout = x_cat.layout
            x_cat_memory_config = x_cat.memory_config()
            torch_x_cat = torch.load("torch_detect_22._inference.concat.pt").reshape(tuple(x_cat.shape))
            ttnn.deallocate(x_cat)
            x_cat = ttnn.from_torch(
                torch_x_cat,
                dtype=x_cat_dtype,
                layout=x_cat_layout,
                memory_config=x_cat_memory_config,
                device=self.device,
            )

        box = ttnn.slice(x_cat, [0, 0, 0], [1, 64, x_cat.shape[2]], memory_config=ttnn.L1_MEMORY_CONFIG)
        cls = ttnn.slice(x_cat, [0, 64, 0], [1, 144, x_cat.shape[2]], memory_config=ttnn.L1_MEMORY_CONFIG)
        if save_pt:
            torch_box = torch.Tensor(ttnn.to_torch(ttnn.from_device(box))).to(torch.float32)  # [1, 64, 8400]
            torch_cls = torch.Tensor(ttnn.to_torch(ttnn.from_device(cls))).to(torch.float32)  # [1, 80, 8400]
            torch.save(torch.Tensor(torch_box), "ttnn_detect_22._inference.box.pt")
            torch.save(torch.Tensor(torch_cls), "ttnn_detect_22._inference.cls.pt")
        if regen_input:
            box_dtype = box.dtype
            box_layout = box.layout
            box_memory_config = box.memory_config()
            torch_box = torch.load("torch_detect_22._inference.box.pt").reshape(tuple(box.shape))
            ttnn.deallocate(box)
            box = ttnn.from_torch(
                torch_box, dtype=box_dtype, layout=box_layout, memory_config=box_memory_config, device=self.device
            )
            cls_dtype = cls.dtype
            cls_layout = cls.layout
            cls_memory_config = cls.memory_config()
            torch_cls = torch.load("torch_detect_22._inference.cls.pt").reshape(tuple(cls.shape))
            ttnn.deallocate(cls)
            cls = ttnn.from_torch(
                torch_cls, dtype=cls_dtype, layout=cls_layout, memory_config=cls_memory_config, device=self.device
            )
        if use_signpost:
            signpost(header="Detect - dfl")
        dfl = self.dfl_module(box)
        if save_pt:
            torch_dfl = torch.Tensor(ttnn.to_torch(ttnn.from_device(dfl))).to(torch.float32)  # [1, 4, 8400]
            torch.save(torch.Tensor(torch_dfl), "ttnn_detect_22._inference.dfl.pt")
        if regen_input:
            dfl_dtype = dfl.dtype
            dfl_layout = dfl.layout
            dfl_memory_config = dfl.memory_config()
            torch_dfl = torch.load("torch_detect_22._inference.dfl.pt").reshape(tuple(dfl.shape))
            ttnn.deallocate(dfl)
            dfl = ttnn.from_torch(
                torch_dfl, dtype=dfl_dtype, layout=dfl_layout, memory_config=dfl_memory_config, device=self.device
            )

        anchors = ttnn.to_memory_config(anchors, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        if use_signpost:
            signpost(header="ttnn_decode_bboxes")
        dbox = ttnn_decode_bboxes(self.device, dfl, anchors, save_pt=save_pt, regen_input=regen_input)
        if save_pt:
            torch_dbox = torch.Tensor(ttnn.to_torch(ttnn.from_device(dbox))).to(torch.float32)  # [1, 4, 2100]
            torch.save(torch.Tensor(torch_dbox), "ttnn_detect_22._inference.dbox.pt")
        if regen_input:
            dbox_dtype = dbox.dtype
            dbox_layout = dbox.layout
            dbox_memory_config = dbox.memory_config()
            torch_dbox = torch.load("torch_detect_22._inference.dbox.pt").reshape(tuple(dbox.shape))
            ttnn.deallocate(dbox)
            dbox = ttnn.from_torch(
                torch_dbox, dtype=dbox_dtype, layout=dbox_layout, memory_config=dbox_memory_config, device=self.device
            )
        # dbox = ttnn.to_dtype(dbox, dtype=ttnn.bfloat8_b)  # Fail: to_dtype only supports host tensors
        strides = ttnn.to_memory_config(strides, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        dbox = ttnn.to_layout(dbox, ttnn.TILE_LAYOUT)
        dbox = ttnn.multiply(dbox, strides, dtype=ttnn.bfloat8_b)
        cls = ttnn.sigmoid(cls)
        if save_pt:
            torch_dbox = torch.Tensor(ttnn.to_torch(ttnn.from_device(dbox))).to(torch.float32)  # [1, 4, 8400]
            torch.save(torch.Tensor(torch_dbox), "ttnn_detect_22._inference.dbox_strides.pt")
            torch_cls = torch.Tensor(ttnn.to_torch(ttnn.from_device(cls))).to(torch.float32)  # [1, 4, 8400]
            torch.save(torch.Tensor(torch_cls), "ttnn_detect_22._inference.cls_sigmoid.pt")

        return [ttnn.concat((dbox, cls), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG), x]


class TtDetectionModel:
    def __init__(self, device, parameters, res=(640, 640), batch_size=1, reg_max=16):
        self.device = device
        self.parameters = parameters
        self.res = res
        self.reg_max = reg_max
        self.batch_size = batch_size

        self.conv_0 = TtConv(
            device,
            parameters,
            "model.0",
            input_params=conv_config["input_params"][0],
            act_block_h=False,
            enable_split_reader=True,
            deallocate_activation=True,
        )
        self.conv_1 = TtConv(
            device,
            parameters,
            "model.1",
            input_params=conv_config["input_params"][1],
            act_block_h=False,
            enable_split_reader=True,
            block_shard=True,
        )
        self.c2f_2 = TtC2f(
            device,
            parameters,
            "model.2",
            n=1,
            shortcut=True,
            input_params=c2f_configs["model.2"]["input_params"],
        )
        self.conv_3 = TtConv(
            device,
            parameters,
            "model.3",
            input_params=conv_config["input_params"][2],
            deallocate_activation=False,
        )
        self.c2f_4 = TtC2f(
            device,
            parameters,
            "model.4",
            n=2,
            shortcut=True,
            input_params=c2f_configs["model.4"]["input_params"],
        )
        self.conv_5 = TtConv(device, parameters, "model.5", input_params=[3, 2, 1, 256, 128], block_shard=True)
        self.c2f_6 = TtC2f(
            device,
            parameters,
            "model.6",
            n=2,
            shortcut=True,
            block_shard=True,
            change_shard=True,
            input_params=c2f_configs["model.6"]["input_params"],
        )
        self.conv_7 = TtConv(
            device,
            parameters,
            "model.7",
            input_params=conv_config["input_params"][3],
            block_shard=True,
            reshard_if_not_optimal=False,
        )
        self.c2f_8 = TtC2f(
            device,
            parameters,
            "model.8",
            n=1,
            shortcut=True,
            change_shard=True,
            block_shard=True,
            input_params=c2f_configs["model.8"]["input_params"],
        )
        self.sppf_9 = TtSppf(
            device, parameters, "model.9", input_params=sppf_configs["input_params"], batch_size=self.batch_size
        )
        self.c2f_12 = TtC2f(
            device,
            parameters,
            "model.12",
            n=1,
            shortcut=False,
            bfloat8=True,
            block_shard=True,
            input_params=c2f_configs["model.12"]["input_params"],
        )
        self.c2f_15 = TtC2f(
            device,
            parameters,
            "model.15",
            n=1,
            shortcut=False,
            input_params=c2f_configs["model.15"]["input_params"],
        )
        self.conv_16 = TtConv(
            device, parameters, "model.16", input_params=conv_config["input_params"][4], block_shard=True
        )
        self.c2f_18 = TtC2f(
            device,
            parameters,
            "model.18",
            n=1,
            shortcut=False,
            input_params=c2f_configs["model.18"]["input_params"],
        )
        self.conv_19 = TtConv(
            device, parameters, "model.19", input_params=conv_config["input_params"][5], block_shard=True
        )
        self.c2f_21 = TtC2f(
            device,
            parameters,
            "model.21",
            n=1,
            shortcut=False,
            input_params=c2f_configs["model.21"]["input_params"],
            block_shard=True,
            change_shard=False,
        )
        self.detect_22 = TtDetect(device, parameters, "model.22", detect_config)

    def __call__(self, x, _save_pt=False, _regen_input=False):
        # x should be a 4D tensor of shape (1, 1, N * H * W, C)
        global N
        H, W = self.res
        N = int(x.shape[-2] / (H * W))

        global save_pt, regen_input
        save_pt = _save_pt
        regen_input = _regen_input

        if save_pt:
            torch_im = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_im = torch_im[:, :, :, :3].reshape([-1, H, W, 3]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_im), "ttnn_im.pt")

        print(f"conv_0 input shape: {x.shape}, out_h: {H}, out_w: {W}")
        if use_signpost:
            signpost(header="start conv_0")
        conv_0, out_h, out_w = self.conv_0(x)
        print(f"conv_0 output shape: {conv_0.shape}, out_h: {out_h}, out_w: {out_w}")
        if save_pt:
            torch_conv_0 = torch.Tensor(ttnn.to_torch(ttnn.from_device(conv_0)))
            torch_conv_0 = torch_conv_0.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_conv_0), "ttnn_conv_0.pt")
        if regen_input:
            conv_0_dtype = conv_0.dtype
            conv_0_layout = conv_0.layout
            conv_0_memory_config = conv_0.memory_config()
            torch_conv_0 = torch.load("torch_conv_0.pt").permute(0, 2, 3, 1).reshape(tuple(conv_0.shape))
            ttnn.deallocate(conv_0)
            conv_0 = ttnn.from_torch(
                torch_conv_0,
                dtype=conv_0_dtype,
                layout=conv_0_layout,
                memory_config=conv_0_memory_config,
                device=self.device,
            )

        if use_signpost:
            signpost(header="start conv_1")
        conv_1, out_h, out_w = self.conv_1(conv_0)
        ttnn.deallocate(conv_0)
        if save_pt:
            torch_conv_1 = torch.Tensor(ttnn.to_torch(ttnn.from_device(conv_1)))
            torch_conv_1 = torch_conv_1.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_conv_1), "ttnn_conv_1.pt")
        if regen_input:
            conv_1_dtype = conv_1.dtype
            conv_1_layout = conv_1.layout
            conv_1_memory_config = conv_1.memory_config()
            torch_conv_1 = torch.load("torch_conv_1.pt").permute(0, 2, 3, 1).reshape(tuple(conv_1.shape))
            ttnn.deallocate(conv_1)
            conv_1 = ttnn.from_torch(
                torch_conv_1,
                dtype=conv_1_dtype,
                layout=conv_1_layout,
                memory_config=conv_1_memory_config,
                device=self.device,
            )

        if use_signpost:
            signpost(header="start c2f_2")

        c2f_2, out_h, out_w = self.c2f_2(conv_1, reshard_bottleneck_input=True)
        ttnn.deallocate(conv_1)
        if save_pt:
            torch_c2f_2 = torch.Tensor(ttnn.to_torch(ttnn.from_device(c2f_2)))
            torch_c2f_2 = torch_c2f_2.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_c2f_2), "ttnn_c2f_2.pt")
        if regen_input:
            c2f_2_dtype = c2f_2.dtype
            c2f_2_layout = c2f_2.layout
            c2f_2_memory_config = c2f_2.memory_config()
            torch_c2f_2 = torch.load("torch_c2f_2.pt").permute(0, 2, 3, 1).reshape(tuple(c2f_2.shape))
            ttnn.deallocate(c2f_2)
            c2f_2 = ttnn.from_torch(
                torch_c2f_2,
                dtype=c2f_2_dtype,
                layout=c2f_2_layout,
                memory_config=c2f_2_memory_config,
                device=self.device,
            )

        if use_signpost:
            signpost(header="start conv_3")
        conv_3, out_h, out_w = self.conv_3(c2f_2)
        ttnn.deallocate(c2f_2)
        if save_pt:
            torch_conv_3 = torch.Tensor(ttnn.to_torch(ttnn.from_device(conv_3)))
            torch_conv_3 = torch_conv_3.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_conv_3), "ttnn_conv_3.pt")
        if regen_input:
            conv_3_dtype = conv_3.dtype
            conv_3_layout = conv_3.layout
            conv_3_memory_config = conv_3.memory_config()
            torch_conv_3 = torch.load("torch_conv_3.pt").permute(0, 2, 3, 1).reshape(tuple(conv_3.shape))
            ttnn.deallocate(conv_3)
            conv_3 = ttnn.from_torch(
                torch_conv_3,
                dtype=conv_3_dtype,
                layout=conv_3_layout,
                memory_config=conv_3_memory_config,
                device=self.device,
            )

        if use_signpost:
            signpost(header="start c2f_4")
        c2f_4, out_h, out_w = self.c2f_4(conv_3)
        ttnn.deallocate(conv_3)
        if save_pt:
            torch_c2f_4 = torch.Tensor(ttnn.to_torch(ttnn.from_device(c2f_4)))
            torch_c2f_4 = torch_c2f_4.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_c2f_4), "ttnn_c2f_4.pt")
        if regen_input:
            c2f_4_dtype = c2f_4.dtype
            c2f_4_layout = c2f_4.layout
            c2f_4_memory_config = c2f_4.memory_config()
            torch_c2f_4 = torch.load("torch_c2f_4.pt").permute(0, 2, 3, 1).reshape(tuple(c2f_4.shape))
            ttnn.deallocate(c2f_4)
            c2f_4 = ttnn.from_torch(
                torch_c2f_4,
                dtype=c2f_4_dtype,
                layout=c2f_4_layout,
                memory_config=c2f_4_memory_config,
                device=self.device,
            )
        # c2f_4 = ttnn.sharded_to_interleaved(c2f_4, ttnn.L1_MEMORY_CONFIG)
        # c2f_4 = ttnn.reallocate(c2f_4, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if use_signpost:
            signpost(header="start conv_5")
        conv_5, out_h, out_w = self.conv_5(c2f_4)
        if save_pt:
            torch_conv_5 = torch.Tensor(ttnn.to_torch(ttnn.from_device(conv_5)))
            torch_conv_5 = torch_conv_5.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_conv_5), "ttnn_conv_5.pt")
        if regen_input:
            conv_5_dtype = conv_5.dtype
            conv_5_layout = conv_5.layout
            conv_5_memory_config = conv_5.memory_config()
            torch_conv_5 = torch.load("torch_conv_5.pt").permute(0, 2, 3, 1).reshape(tuple(conv_5.shape))
            ttnn.deallocate(conv_5)
            conv_5 = ttnn.from_torch(
                torch_conv_5,
                dtype=conv_5_dtype,
                layout=conv_5_layout,
                memory_config=conv_5_memory_config,
                device=self.device,
            )

        if use_signpost:
            signpost(header="start c2f_6")
        c2f_6, out_h, out_w = self.c2f_6(conv_5)
        ttnn.deallocate(conv_5)
        if save_pt:
            torch_c2f_6 = torch.Tensor(ttnn.to_torch(ttnn.from_device(c2f_6)))
            torch_c2f_6 = torch_c2f_6.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_c2f_6), "ttnn_c2f_6.pt")
        if regen_input:
            c2f_6_dtype = c2f_6.dtype
            c2f_6_layout = c2f_6.layout
            c2f_6_memory_config = c2f_6.memory_config()
            torch_c2f_6 = torch.load("torch_c2f_6.pt").permute(0, 2, 3, 1).reshape(tuple(c2f_6.shape))
            ttnn.deallocate(c2f_6)
            c2f_6 = ttnn.from_torch(
                torch_c2f_6,
                dtype=c2f_6_dtype,
                layout=c2f_6_layout,
                memory_config=c2f_6_memory_config,
                device=self.device,
            )
        # c2f_6 = ttnn.reallocate(c2f_6, memory_config=ttnn.L1_MEMORY_CONFIG)
        c2f_6 = ttnn.sharded_to_interleaved(c2f_6, ttnn.L1_MEMORY_CONFIG)

        if use_signpost:
            signpost(header="start conv_7")
        conv_7, out_h, out_w = self.conv_7(c2f_6)
        if save_pt:
            torch_conv_7 = torch.Tensor(ttnn.to_torch(ttnn.from_device(conv_7)))
            torch_conv_7 = torch_conv_7.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_conv_7), "ttnn_conv_7.pt")
        if regen_input:
            conv_7_dtype = conv_7.dtype
            conv_7_layout = conv_7.layout
            conv_7_memory_config = conv_7.memory_config()
            torch_conv_7 = torch.load("torch_conv_7.pt").permute(0, 2, 3, 1).reshape(tuple(conv_7.shape))
            ttnn.deallocate(conv_7)
            conv_7 = ttnn.from_torch(
                torch_conv_7,
                dtype=conv_7_dtype,
                layout=conv_7_layout,
                memory_config=conv_7_memory_config,
                device=self.device,
            )
        # conv_7 = ttnn.sharded_to_interleaved(conv_7, ttnn.L1_MEMORY_CONFIG)

        if use_signpost:
            signpost(header="start c2f_8")
        c2f_8, out_h, out_w = self.c2f_8(conv_7, reshard_bottleneck_input=True)
        ttnn.deallocate(conv_7)
        if save_pt:
            torch_c2f_8 = torch.Tensor(ttnn.to_torch(ttnn.from_device(c2f_8)))
            torch_c2f_8 = torch_c2f_8.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_c2f_8), "ttnn_c2f_8.pt")
        if regen_input:
            c2f_8_dtype = c2f_8.dtype
            c2f_8_layout = c2f_8.layout
            c2f_8_memory_config = c2f_8.memory_config()
            torch_c2f_8 = torch.load("torch_c2f_8.pt").permute(0, 2, 3, 1).reshape(tuple(c2f_8.shape))
            ttnn.deallocate(c2f_8)
            c2f_8 = ttnn.from_torch(
                torch_c2f_8,
                dtype=c2f_8_dtype,
                layout=c2f_8_layout,
                memory_config=c2f_8_memory_config,
                device=self.device,
            )
        # c2f_8 = ttnn.sharded_to_interleaved(c2f_8, ttnn.L1_MEMORY_CONFIG)

        if use_signpost:
            signpost(header="start sppf_9")
        nine, out_h, out_w = self.sppf_9(c2f_8)
        ttnn.deallocate(c2f_8)
        if save_pt:
            torch_sppf_9 = torch.Tensor(ttnn.to_torch(ttnn.from_device(nine)))
            torch_sppf_9 = torch_sppf_9.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_sppf_9), "ttnn_sppf_9.pt")
        if regen_input:
            nine_dtype = nine.dtype
            nine_layout = nine.layout
            nine_memory_config = nine.memory_config()
            torch_sppf_9 = torch.load("torch_sppf_9.pt").permute(0, 2, 3, 1).reshape(tuple(nine.shape))
            ttnn.deallocate(nine)
            nine = ttnn.from_torch(
                torch_sppf_9, dtype=nine_dtype, layout=nine_layout, memory_config=nine_memory_config, device=self.device
            )

        # nine = ttnn.to_memory_config(nine, ttnn.L1_MEMORY_CONFIG)
        nine = ttnn.sharded_to_interleaved(nine, ttnn.L1_MEMORY_CONFIG)
        sppf_9 = ttnn.to_layout(nine, ttnn.ROW_MAJOR_LAYOUT)  # before upsample_10 : pcc broken
        nine = ttnn.to_layout(nine, ttnn.ROW_MAJOR_LAYOUT)  # before concat_20 : pcc is recovered ???
        # same operation to_layout has broken pcc, but it is recovered the 2nd time
        # I don't know why, but it works! (v0.59.0 & A0)
        # if save_pt:
        #     torch_sppf_9 = torch.Tensor(ttnn.to_torch(ttnn.from_device(sppf_9)))
        #     torch_sppf_9 = torch_sppf_9.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
        #     torch.save(torch.Tensor(torch_sppf_9), "ttnn_sppf_9.row.pt")
        #     torch_nine = torch.Tensor(ttnn.to_torch(ttnn.from_device(nine)))
        #     torch_nine = torch_nine.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
        #     torch.save(torch.Tensor(torch_nine), "ttnn_sppf_9.nine.pt")

        sppf_9 = ttnn.reshape(nine, (self.batch_size, out_h, out_w, nine.shape[-1]))

        nhw = sppf_9.shape[0] * sppf_9.shape[1] * sppf_9.shape[2]
        num_cores = determine_num_cores(nhw, sppf_9.shape[2], max_cores=20)
        core_grid = get_core_grid_from_num_cores(num_cores, grid_rows=4, grid_cols=5)  # cols=x, rows=y
        shardspec = ttnn.create_sharded_memory_config_(
            sppf_9.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )
        if sppf_9.is_sharded():
            sppf_9 = ttnn.reshard(sppf_9, shardspec)
        else:
            sppf_9 = ttnn.interleaved_to_sharded(sppf_9, shardspec)

        if use_signpost:
            signpost(header="start upsample_10")
        sppf_9 = ttnn.upsample(sppf_9, (2, 2), memory_config=sppf_9.memory_config())
        if save_pt:
            torch_upsample_10 = torch.Tensor(ttnn.to_torch(ttnn.from_device(sppf_9)))
            torch_upsample_10 = (
                torch_upsample_10.reshape([N, out_h * 2, out_w * 2, -1]).permute(0, 3, 1, 2).to(torch.float32)
            )
            torch.save(torch.Tensor(torch_upsample_10), "ttnn_upsample_10.pt")
        if regen_input:
            sppf_9_dtype = sppf_9.dtype
            sppf_9_layout = sppf_9.layout
            sppf_9_memory_config = sppf_9.memory_config()
            torch_upsample_10 = torch.load("torch_upsample_10.pt").permute(0, 2, 3, 1).reshape(tuple(sppf_9.shape))
            ttnn.deallocate(sppf_9)
            sppf_9 = ttnn.from_torch(
                torch_upsample_10,
                dtype=sppf_9_dtype,
                layout=sppf_9_layout,
                memory_config=sppf_9_memory_config,
                device=self.device,
            )

        x = ttnn.reshape(sppf_9, (1, 1, (self.batch_size) * sppf_9.shape[1] * sppf_9.shape[2], sppf_9.shape[-1]))

        if use_signpost:
            signpost(header="start concat_11")
        c2f_6 = ttnn.to_layout(c2f_6, layout=ttnn.ROW_MAJOR_LAYOUT)
        c2f_6 = ttnn.to_memory_config(c2f_6, ttnn.L1_MEMORY_CONFIG)
        x = sharded_concat([x, c2f_6])
        if save_pt:
            torch_concat_11 = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_concat_11 = (
                torch_concat_11.reshape([N, out_h * 2, out_w * 2, -1]).permute(0, 3, 1, 2).to(torch.float32)
            )
            torch.save(torch.Tensor(torch_concat_11), "ttnn_concat_11.pt")
        if regen_input:
            x_dtype = x.dtype
            x_layout = x.layout
            x_memory_config = x.memory_config()
            torch_concat_11 = torch.load("torch_concat_11.pt").permute(0, 2, 3, 1).reshape(tuple(x.shape))
            ttnn.deallocate(x)
            x = ttnn.from_torch(
                torch_concat_11, dtype=x_dtype, layout=x_layout, memory_config=x_memory_config, device=self.device
            )
        ttnn.deallocate(c2f_6)

        if use_signpost:
            signpost(header="start c2f_12")
        c2f_12, out_h, out_w = self.c2f_12(x)
        ttnn.deallocate(x)
        ttnn.deallocate(sppf_9)
        if save_pt:
            torch_c2f_12 = torch.Tensor(ttnn.to_torch(ttnn.from_device(c2f_12)))
            torch_c2f_12 = torch_c2f_12.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_c2f_12), "ttnn_c2f_12.pt")
        if regen_input:
            c2f_12_dtype = c2f_12.dtype
            c2f_12_layout = c2f_12.layout
            c2f_12_memory_config = c2f_12.memory_config()
            torch_c2f_12 = torch.load("torch_c2f_12.pt").permute(0, 2, 3, 1).reshape(tuple(c2f_12.shape))
            ttnn.deallocate(c2f_12)
            c2f_12 = ttnn.from_torch(
                torch_c2f_12,
                dtype=c2f_12_dtype,
                layout=c2f_12_layout,
                memory_config=c2f_12_memory_config,
                device=self.device,
            )

        if use_signpost:
            signpost(header="start c2f_12 post-processing")
        c2f_12 = ttnn.sharded_to_interleaved(c2f_12, ttnn.L1_MEMORY_CONFIG)
        c2f_12 = ttnn.to_layout(c2f_12, ttnn.ROW_MAJOR_LAYOUT)
        twelve = ttnn.clone(c2f_12, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        c2f_12 = ttnn.reshape(
            c2f_12, (self.batch_size, out_h, out_w, c2f_12.shape[-1]), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        nhw = c2f_12.shape[0] * c2f_12.shape[1] * c2f_12.shape[2]
        num_cores = determine_num_cores(nhw, c2f_12.shape[2], max_cores=20)
        core_grid = get_core_grid_from_num_cores(num_cores, grid_rows=4, grid_cols=5)  # cols=x, rows=y
        shardspec = ttnn.create_sharded_memory_config_(
            c2f_12.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )
        if c2f_12.is_sharded():
            c2f_12 = ttnn.reshard(c2f_12, shardspec)
        else:
            c2f_12 = ttnn.interleaved_to_sharded(c2f_12, shardspec)

        if use_signpost:
            signpost(header="start upsample_13")
        c2f_12 = ttnn.upsample(c2f_12, (2, 2), memory_config=c2f_12.memory_config())
        if save_pt:
            torch_upsample_13 = torch.Tensor(ttnn.to_torch(ttnn.from_device(c2f_12)))
            torch_upsample_13 = (
                torch_upsample_13.reshape([N, out_h * 2, out_w * 2, -1]).permute(0, 3, 1, 2).to(torch.float32)
            )
            torch.save(torch.Tensor(torch_upsample_13), "ttnn_upsample_13.pt")
        if regen_input:
            c2f_12_dtype = c2f_12.dtype
            c2f_12_layout = c2f_12.layout
            c2f_12_memory_config = c2f_12.memory_config()
            torch_upsample_13 = torch.load("torch_upsample_13.pt").permute(0, 2, 3, 1).reshape(tuple(c2f_12.shape))
            ttnn.deallocate(c2f_12)
            c2f_12 = ttnn.from_torch(
                torch_upsample_13,
                dtype=c2f_12_dtype,
                layout=c2f_12_layout,
                memory_config=c2f_12_memory_config,
                device=self.device,
            )

        if use_signpost:
            signpost(header="start concat_14")
        x = ttnn.reshape(c2f_12, (1, 1, (self.batch_size) * c2f_12.shape[1] * c2f_12.shape[2], c2f_12.shape[-1]))
        c2f_4 = ttnn.to_layout(c2f_4, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = sharded_concat([x, c2f_4])
        ttnn.deallocate(c2f_4)
        ttnn.deallocate(c2f_12)
        if save_pt:
            torch_concat_14 = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_concat_14 = (
                torch_concat_14.reshape([N, out_h * 2, out_w * 2, -1]).permute(0, 3, 1, 2).to(torch.float32)
            )
            torch.save(torch.Tensor(torch_concat_14), "ttnn_concat_14.pt")
        if regen_input:
            x_dtype = x.dtype
            x_layout = x.layout
            x_memory_config = x.memory_config()
            torch_concat_14 = torch.load("torch_concat_14.pt").permute(0, 2, 3, 1).reshape(tuple(x.shape))
            ttnn.deallocate(x)
            x = ttnn.from_torch(
                torch_concat_14, dtype=x_dtype, layout=x_layout, memory_config=x_memory_config, device=self.device
            )

        if use_signpost:
            signpost(header="start c2f_15")
        c2f_15, out_h, out_w = self.c2f_15(x)
        ttnn.deallocate(x)
        if save_pt:
            torch_c2f_15 = torch.Tensor(ttnn.to_torch(ttnn.from_device(c2f_15)))
            torch_c2f_15 = torch_c2f_15.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_c2f_15), "ttnn_c2f_15.pt")
        if regen_input:
            c2f_15_dtype = c2f_15.dtype
            c2f_15_layout = c2f_15.layout
            c2f_15_memory_config = c2f_15.memory_config()
            torch_c2f_15 = torch.load("torch_c2f_15.pt").permute(0, 2, 3, 1).reshape(tuple(c2f_15.shape))
            ttnn.deallocate(c2f_15)
            c2f_15 = ttnn.from_torch(
                torch_c2f_15,
                dtype=c2f_15_dtype,
                layout=c2f_15_layout,
                memory_config=c2f_15_memory_config,
                device=self.device,
            )

        if use_signpost:
            signpost(header="start conv_16")
        c2f_15 = ttnn.sharded_to_interleaved(c2f_15, ttnn.L1_MEMORY_CONFIG)
        fifteen = ttnn.clone(c2f_15, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        fifteen_h, fifteen_w = out_h, out_w
        conv_16, out_h, out_w = self.conv_16(c2f_15)
        ttnn.deallocate(c2f_15)
        if save_pt:
            torch_conv_16 = torch.Tensor(ttnn.to_torch(ttnn.from_device(conv_16)))
            torch_conv_16 = torch_conv_16.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_conv_16), "ttnn_conv_16.pt")
        if regen_input:
            conv_16_dtype = conv_16.dtype
            conv_16_layout = conv_16.layout
            conv_16_memory_config = conv_16.memory_config()
            torch_conv_16 = torch.load("torch_conv_16.pt").permute(0, 2, 3, 1).reshape(tuple(conv_16.shape))
            ttnn.deallocate(conv_16)
            conv_16 = ttnn.from_torch(
                torch_conv_16,
                dtype=conv_16_dtype,
                layout=conv_16_layout,
                memory_config=conv_16_memory_config,
                device=self.device,
            )

        if use_signpost:
            signpost(header="start concat_17")
        conv_16 = ttnn.sharded_to_interleaved(conv_16, ttnn.L1_MEMORY_CONFIG)
        conv_16 = ttnn.to_layout(conv_16, ttnn.ROW_MAJOR_LAYOUT)
        x = sharded_concat([conv_16, twelve])
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(twelve)
        ttnn.deallocate(conv_16)
        if save_pt:
            torch_concat_17 = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_concat_17 = torch_concat_17.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_concat_17), "ttnn_concat_17.pt")
        if regen_input:
            x_dtype = x.dtype
            x_layout = x.layout
            x_memory_config = x.memory_config()
            torch_concat_17 = torch.load("torch_concat_17.pt").permute(0, 2, 3, 1).reshape(tuple(x.shape))
            ttnn.deallocate(x)
            x = ttnn.from_torch(
                torch_concat_17, dtype=x_dtype, layout=x_layout, memory_config=x_memory_config, device=self.device
            )

        if use_signpost:
            signpost(header="start c2f_18")
        c2f_18, out_h, out_w = self.c2f_18(x)
        ttnn.deallocate(x)
        if save_pt:
            torch_c2f_18 = torch.Tensor(ttnn.to_torch(ttnn.from_device(c2f_18)))
            torch_c2f_18 = torch_c2f_18.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_c2f_18), "ttnn_c2f_18.pt")
        if regen_input:
            c2f_18_dtype = c2f_18.dtype
            c2f_18_layout = c2f_18.layout
            c2f_18_memory_config = c2f_18.memory_config()
            torch_c2f_18 = torch.load("torch_c2f_18.pt").permute(0, 2, 3, 1).reshape(tuple(c2f_18.shape))
            ttnn.deallocate(c2f_18)
            c2f_18 = ttnn.from_torch(
                torch_c2f_18,
                dtype=c2f_18_dtype,
                layout=c2f_18_layout,
                memory_config=c2f_18_memory_config,
                device=self.device,
            )

        if use_signpost:
            signpost(header="start conv_19")
        c2f_18 = ttnn.sharded_to_interleaved(c2f_18, ttnn.L1_MEMORY_CONFIG)
        eighteen = ttnn.clone(c2f_18, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        eighteen_h, eighteen_w = out_h, out_w
        conv_19, out_h, out_w = self.conv_19(c2f_18)
        ttnn.deallocate(c2f_18)
        if save_pt:
            torch_conv_19 = torch.Tensor(ttnn.to_torch(ttnn.from_device(conv_19)))
            torch_conv_19 = torch_conv_19.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_conv_19), "ttnn_conv_19.pt")
        if regen_input:
            conv_19_dtype = conv_19.dtype
            conv_19_layout = conv_19.layout
            conv_19_memory_config = conv_19.memory_config()
            torch_conv_19 = torch.load("torch_conv_19.pt").permute(0, 2, 3, 1).reshape(tuple(conv_19.shape))
            ttnn.deallocate(conv_19)
            conv_19 = ttnn.from_torch(
                torch_conv_19,
                dtype=conv_19_dtype,
                layout=conv_19_layout,
                memory_config=conv_19_memory_config,
                device=self.device,
            )

        if use_signpost:
            signpost(header="start concat_20")
        conv_19 = ttnn.sharded_to_interleaved(conv_19, ttnn.L1_MEMORY_CONFIG)
        conv_19 = ttnn.to_layout(conv_19, ttnn.ROW_MAJOR_LAYOUT)
        # nine = ttnn.to_layout(nine, layout=ttnn.ROW_MAJOR_LAYOUT)
        # if save_pt:
        #     torch_nine = torch.Tensor(ttnn.to_torch(ttnn.from_device(nine)))
        #     torch_nine = torch_nine.reshape([N, 10, 10, -1]).permute(0, 3, 1, 2).to(torch.float32)
        #     torch.save(torch.Tensor(torch_nine), "ttnn_sppf_9.nine.pt")
        x = sharded_concat([conv_19, nine])
        if save_pt:
            torch_concat_20 = torch.Tensor(ttnn.to_torch(ttnn.from_device(x)))
            torch_concat_20 = torch_concat_20.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_concat_20), "ttnn_concat_20.pt")
        if regen_input:
            x_dtype = x.dtype
            x_layout = x.layout
            x_memory_config = x.memory_config()
            torch_concat_20 = torch.load("torch_concat_20.pt").permute(0, 2, 3, 1).reshape(tuple(x.shape))
            ttnn.deallocate(x)
            x = ttnn.from_torch(
                torch_concat_20, dtype=x_dtype, layout=x_layout, memory_config=x_memory_config, device=self.device
            )

        ttnn.deallocate(nine)
        ttnn.deallocate(conv_19)

        if use_signpost:
            signpost(header="start c2f_21")
        c2f_21, out_h, out_w = self.c2f_21(x, reshard_bottleneck_input=True)
        c2f_21 = ttnn.sharded_to_interleaved(c2f_21, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x)
        if save_pt:
            torch_c2f_21 = torch.Tensor(ttnn.to_torch(ttnn.from_device(c2f_21)))
            torch_c2f_21 = torch_c2f_21.reshape([N, out_h, out_w, -1]).permute(0, 3, 1, 2).to(torch.float32)
            torch.save(torch.Tensor(torch_c2f_21), "ttnn_c2f_21.pt")
        if regen_input:
            c2f_21_dtype = c2f_21.dtype
            c2f_21_layout = c2f_21.layout
            c2f_21_memory_config = c2f_21.memory_config()
            torch_c2f_21 = torch.load("torch_c2f_21.pt").permute(0, 2, 3, 1).reshape(tuple(c2f_21.shape))
            ttnn.deallocate(c2f_21)
            c2f_21 = ttnn.from_torch(
                torch_c2f_21,
                dtype=c2f_21_dtype,
                layout=c2f_21_layout,
                memory_config=c2f_21_memory_config,
                device=self.device,
            )
        x = [fifteen, eighteen, c2f_21]

        if use_signpost:
            signpost(header="start detect_22")
        if save_pt:
            torch_detect_22_in = []
            for _x, _hw in zip(x, [(fifteen_h, fifteen_w), (eighteen_h, eighteen_w), (out_h, out_w)]):
                _h, _w = _hw
                torch_x = torch.Tensor(ttnn.to_torch(ttnn.from_device(_x)))
                torch_x = torch_x.reshape([N, _h, _w, -1]).permute(0, 3, 1, 2).to(torch.float32)
                torch_detect_22_in.append(torch_x)
            torch.save(torch_detect_22_in, "ttnn_detect_22.in.pt")
        x = self.detect_22(x, nc=80, ch=(128, 256, 512), reg_max=self.reg_max)
        if save_pt:
            torch_detect_22_y = torch.Tensor(ttnn.to_torch(ttnn.from_device(x[0]))).to(torch.float32)
            torch.save(torch.Tensor(torch_detect_22_y), "ttnn_detect_22.y.pt")
            torch.save(torch_detect_22_y[:, :4, :], "ttnn_detect_22.y.0.pt")
            torch.save(torch_detect_22_y[:, 4:, :], "ttnn_detect_22.y.1.pt")
        return x


class TtYolov8sModel:
    def __init__(self, device, parameters, res=(320, 320), batch_size=1):
        self.device = device
        self.parameters = parameters
        self.res = res
        self.detection_model = TtDetectionModel(device, parameters, res, batch_size)
        self.input_tensor = None
        self.output_tensor = None

    def __call__(self, x=None, _save_pt=False, _regen_input=False):
        self.input_tensor = x if x is not None else self.input_tensor
        self.output_tensor = self.detection_model(self.input_tensor, _save_pt=_save_pt, _regen_input=_regen_input)
        return self.output_tensor
        # return self.detection_model(x, _save_pt=_save_pt, _regen_input=_regen_input)
