import math

import ttnn
from models.bos_model.pdl.tt.model_processing import sharded_concat, upsample_multicore_common

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TtConv2d:
    def __init__(
        self,
        device,
        parameters,
        conv_pt,
        *,
        has_bias=True,
        act_block_h=32,
        reshard=False,
        deallocate=False,
        activation="relu",
        shard_type="HS",
        dtype=ttnn.bfloat16,
        slice_type=None,
        num_slices=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
    ) -> None:
        self.weights = parameters.conv.weight

        self.conv_pt = conv_pt
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = parameters.conv.bias

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.stride = conv_pt.stride
        self.padding = conv_pt.padding
        self.in_channels = conv_pt.in_channels
        self.out_channels = conv_pt.out_channels
        self.dilation = conv_pt.dilation
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.dtype = dtype
        self.slice_type = slice_type
        self.num_slices = num_slices
        self.device = device
        self.memory_config = memory_config
        self.output_layout = output_layout

        if shard_type == "WS":
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        elif shard_type == "HS":
            self.shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif shard_type == "BS":
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        else:
            self.shard_layout = None

        self.deallocate = deallocate
        self.activation = activation

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape if self.has_bias else ''} {self.kernel_size}"

    def __call__(self, x, inp_h, inp_w):
        if self.slice_type is not None and self.num_slices is not None:
            slice_config = ttnn.Conv2dSliceConfig(
                slice_type=self.slice_type,
                num_slices=self.num_slices,
            )
        else:
            slice_config = None
        conv_config = ttnn.Conv2dConfig(
            # dtype=self.dtype,
            weights_dtype=self.dtype,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate,
            activation=self.activation,
            reshard_if_not_optimal=self.reshard,
            output_layout=self.output_layout,
            reallocate_halo_output=True,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,  # PCC drop for few cases
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [x, [out_h, out_w], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            bias_tensor=self.bias if self.has_bias else None,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=1,
            input_height=inp_h,
            input_width=inp_w,
            conv_config=conv_config,
            compute_config=compute_config,
            slice_config=slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            memory_config=self.memory_config,
        )
        return x, out_h, out_w


class TtAvgPool2d:
    def __init__(self, device, channels, kernel_size=(16, 32), stride=(1, 1), padding=(0, 0)):
        self.stride = stride
        self.device = device
        self.padding = padding
        self.channels = channels
        self.kernel_size = kernel_size
        self.shard_scheme = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    def __call__(self, x, inp_h, inp_w):
        x = ttnn.avg_pool2d(
            input_tensor=x,
            batch_size=1,
            input_h=inp_h,
            input_w=inp_w,
            channels=self.channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=False,
            divisor_override=None,
            count_include_pad=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            applied_shard_scheme=self.shard_scheme,
        )

        return x


class TtASPP:
    def __init__(self, device, parameters, conv_pt):
        self.convs = []
        # conv 1x1
        self.convs.append(TtConv2d(device, parameters.convs[0], conv_pt=conv_pt.convs[0], memory_config=None))
        # atrous convs
        for i in range(3):  # Dilation replacement
            self.convs.append(
                TtConv2d(device, parameters.convs[i + 1], conv_pt.convs[i + 1], shard_type=None, memory_config=None)
            )

        image_pooling = [
            TtAvgPool2d(device, channels=2048),
            TtConv2d(device, parameters.convs[4][1], conv_pt.convs[4][1], shard_type=None, memory_config=None),
        ]

        self.convs.append(image_pooling)
        self.project = TtConv2d(device, parameters.project, conv_pt.project, memory_config=None)
        self.device = device

    def __call__(self, x, inp_h, inp_w):
        if use_signpost:
            signpost(header="ASPP")

        res = []
        for conv in self.convs:
            if isinstance(conv, list):
                if x.shape[1] != 1:
                    x = ttnn.reshape(x, (1, 1, inp_h * inp_w, x.shape[-1]))
                out = conv[0](x, inp_h, inp_w)
                out = conv[1](out, inp_h, inp_w)[0]
            else:
                out = conv(x, inp_h, inp_w)[0]
            out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
            res.append(out)

        res[-1] = upsample_multicore_common(
            self.device, res[-1], scale_h=inp_h // res[-1].shape[1], scale_w=inp_w // res[-1].shape[2]
        )
        res[-1] = ttnn.reshape(res[-1], (1, 1, inp_h * inp_w, res[-1].shape[-1]))

        res = sharded_concat(res)

        res, out_h, out_w = self.project(res, inp_h, inp_w)
        return res, out_h, out_w


class TtDeepLabV3PlusHead:
    def __init__(self, device, parameters, conv_pt, input_shape, shard_concat_dram=False):
        self.decoder = {}
        self.device = device
        self.in_features = input_shape
        self.shard_concat_dram = shard_concat_dram

        for idx, layer_name in enumerate(input_shape):  # [res2, res3, res5]
            decoder_stage = {}
            if idx == 2:
                project_conv = TtASPP(device, parameters[layer_name].project_conv, conv_pt[layer_name].project_conv)
                fuse_conv = None
            else:
                project_conv = TtConv2d(
                    device, parameters[layer_name].project_conv, conv_pt[layer_name].project_conv, memory_config=None
                )

                fuse_conv = [
                    TtConv2d(
                        device, parameters[layer_name].fuse_conv[0], conv_pt[layer_name].fuse_conv[0], shard_type=None
                    ),
                    TtConv2d(
                        device, parameters[layer_name].fuse_conv[1], conv_pt[layer_name].fuse_conv[1], shard_type=None
                    ),
                ]

            decoder_stage["project_conv"] = project_conv
            decoder_stage["fuse_conv"] = fuse_conv

            self.decoder[layer_name] = decoder_stage

    def __call__(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        if use_signpost:
            signpost(header="DeepLabV3PlusHead")

        for f in self.in_features[::-1]:
            x, inp_h, inp_w = features[f]
            proj_x, out_h, out_w = self.decoder[f]["project_conv"](x, inp_h, inp_w)
            proj_x = ttnn.to_layout(proj_x, ttnn.ROW_MAJOR_LAYOUT)

            if self.decoder[f]["fuse_conv"] is None:
                # This is aspp module
                proj_x = ttnn.reshape(proj_x, (1, out_h, out_w, proj_x.shape[-1]))
                y = proj_x
            else:
                # Upsample y
                y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
                y = upsample_multicore_common(self.device, y, scale_h=out_h // y.shape[1], scale_w=out_w // y.shape[2])

                inp_h, inp_w = y.shape[1], y.shape[2]
                y = ttnn.reshape(y, (1, 1, y.shape[1] * y.shape[2], y.shape[-1]))

                y = sharded_concat([proj_x, y], num_cores=20, use_output_dram=self.shard_concat_dram)
                ttnn.deallocate(proj_x)

                # Fuse Conv
                y, out_h, out_w = self.decoder[f]["fuse_conv"][0](y, inp_h, inp_w)
                if f in ["res2"]:
                    y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
                y, out_h, out_w = self.decoder[f]["fuse_conv"][1](y, out_h, out_w)

                y = ttnn.reshape(y, (1, out_h, out_w, y.shape[-1]))
        return y, out_h, out_w


class TtPanopticDeepLabSemSegHead(TtDeepLabV3PlusHead):
    def __init__(self, device, parameters, conv_pt):
        super().__init__(
            device, parameters.decoder, conv_pt.decoder, input_shape=["res2", "res3", "res5"], shard_concat_dram=True
        )
        self.device = device

        self.head = [
            TtConv2d(device, parameters.head[0], conv_pt=conv_pt.head[0], shard_type=None),
            TtConv2d(device, parameters.head[1], conv_pt=conv_pt.head[1], shard_type=None),
        ]

        self.predictor = TtConv2d(device, parameters.predictor, conv_pt=conv_pt.predictor, activation="")

        self.common_stride = 4

    def __call__(self, features):
        if use_signpost:
            signpost(header="PanopticDeepLabSemSegHead")

        y, out_h, out_w = super().__call__(features)

        y, out_h, out_w = self.head[0](y, out_h, out_w)
        y, out_h, out_w = self.head[1](y, out_h, out_w)

        y, out_h, out_w = self.predictor(y, out_h, out_w)

        y = ttnn.reshape(y, (1, out_h, out_w, y.shape[-1]))
        y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        y = upsample_multicore_common(self.device, y, scale_h=self.common_stride, scale_w=self.common_stride)

        return y, out_h, out_w


class TtPanopticDeepLabInsEmbedHead(TtDeepLabV3PlusHead):
    def __init__(self, device, parameters, conv_pt):
        super().__init__(device, parameters.decoder, conv_pt.decoder, input_shape=["res2", "res3", "res5"])
        self.device = device

        self.center_head = [
            TtConv2d(
                device,
                parameters.center_head[0],
                conv_pt=conv_pt.center_head[0],
            ),
            TtConv2d(
                device,
                parameters.center_head[1],
                conv_pt=conv_pt.center_head[1],
            ),
        ]

        self.center_predictor = TtConv2d(
            device,
            parameters.center_predictor,
            conv_pt=conv_pt.center_predictor,
            activation="",
        )

        self.offset_head = [
            TtConv2d(
                device,
                parameters.offset_head[0],
                conv_pt=conv_pt.offset_head[0],
            ),
            TtConv2d(
                device,
                parameters.offset_head[1],
                conv_pt=conv_pt.offset_head[1],
            ),
        ]

        self.offset_predictor = TtConv2d(
            device,
            parameters.offset_predictor,
            conv_pt=conv_pt.offset_predictor,
            activation="",
        )

        self.common_stride = 4

    def __call__(self, features):
        if use_signpost:
            signpost(header="PanopticDeepLabInsEmbedHead")

        y, y_out_h, y_out_w = super().__call__(features)

        # center
        center, out_h, out_w = self.center_head[0](y, y_out_h, y_out_w)
        center, out_h, out_w = self.center_head[1](center, out_h, out_w)
        center, out_h, out_w = self.center_predictor(center, out_h, out_w)

        center = ttnn.reshape(center, (1, out_h, out_w, center.shape[-1]))
        center = ttnn.to_layout(center, ttnn.ROW_MAJOR_LAYOUT)

        # offset
        offset, out_h, out_w = self.offset_head[0](y, y_out_h, y_out_w)
        offset, out_h, out_w = self.offset_head[1](offset, out_h, out_w)
        offset, out_h, out_w = self.offset_predictor(offset, out_h, out_w)

        offset = ttnn.reshape(offset, (1, out_h, out_w, offset.shape[-1]))
        offset = ttnn.to_layout(offset, ttnn.ROW_MAJOR_LAYOUT)

        center = upsample_multicore_common(self.device, center, scale_h=self.common_stride, scale_w=self.common_stride)
        offset = (
            upsample_multicore_common(self.device, offset, scale_h=self.common_stride, scale_w=self.common_stride)
            * self.common_stride
        )

        return center, offset


class TtMaxPool2d:
    def __init__(self, device, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1]):
        self.device = device

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size

    def __call__(self, x, inp_h, inp_w):
        out_h = (
            math.floor(
                (inp_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0]
            )
            + 1
        )
        out_w = (
            math.floor(
                (inp_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1]
            )
            + 1
        )

        # print(x, x.memory_config())
        # print(x, x.memory_config())
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=1,
            input_h=inp_h,
            input_w=inp_w,
            channels=x.shape[-1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return x, out_h, out_w


class TtDeepLabStem:
    def __init__(self, device, parameters, conv_pt):
        self.conv1 = TtConv2d(device, parameters.conv1, conv_pt=conv_pt.conv1, memory_config=None)

        self.conv2 = TtConv2d(device, parameters.conv2, conv_pt=conv_pt.conv2, memory_config=None)

        self.conv3 = TtConv2d(device, parameters.conv3, conv_pt=conv_pt.conv3, memory_config=None)

        self.max_pool = TtMaxPool2d(device)

    def __call__(self, x, inp_h, inp_w):
        if use_signpost:
            signpost(header="DeepLabStem")

        x, out_h, out_w = self.conv1(x, inp_h, inp_w)
        x, out_h, out_w = self.conv2(x, out_h, out_w)

        # Convert shard mode from Physical to Logical to remove Static buffer
        mem_config = x.memory_config()
        mem_config.shard_spec.mode = ttnn.ShardMode.LOGICAL
        x = ttnn.reshard(x, mem_config)

        x, out_h, out_w = self.conv3(x, out_h, out_w)

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = self.max_pool(x, out_h, out_w)
        return x


class TtBottleneckBlock:
    def __init__(self, device, parameters, conv_pt, shard_type="HS"):
        self.device = device

        if "shortcut" in parameters:
            self.shortcut = TtConv2d(
                device,
                parameters=parameters.shortcut,
                conv_pt=conv_pt.shortcut,
                activation="",
                shard_type=shard_type,
            )
        else:
            self.shortcut = None

        self.conv1 = TtConv2d(device, parameters=parameters.conv1, conv_pt=conv_pt.conv1, shard_type=shard_type)

        self.conv2 = TtConv2d(device, parameters=parameters.conv2, conv_pt=conv_pt.conv2, shard_type=shard_type)

        self.conv3 = TtConv2d(
            device,
            parameters=parameters.conv3,
            conv_pt=conv_pt.conv3,
            activation="",
            shard_type=shard_type,
        )

    def __call__(self, x, inp_h, inp_w, temp=""):
        if use_signpost:
            signpost(header="BottleneckBlock")

        out, out_h, out_w = self.conv1(x, inp_h, inp_w)
        out, out_h, out_w = self.conv2(out, out_h, out_w)

        if out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)

        out, out_h, out_w = self.conv3(out, out_h, out_w)

        if self.shortcut is not None:
            shortcut = self.shortcut(x, inp_h, inp_w)[0]
        else:
            shortcut = x

        if shortcut.is_sharded():
            if out.memory_config().memory_layout == shortcut.memory_config().memory_layout:
                if out.memory_config().shard_spec.grid != shortcut.memory_config().shard_spec.grid:
                    memory_config_req = out.memory_config()
                    memory_config_req.shard_spec.shape = shortcut.memory_config().shard_spec.shape
                    memory_config_req.shard_spec.grid = shortcut.memory_config().shard_spec.grid
                    out = ttnn.reshard(out, memory_config_req)
            else:
                out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
                shortcut = ttnn.sharded_to_interleaved(shortcut, ttnn.DRAM_MEMORY_CONFIG)

        out = ttnn.add(out, shortcut, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # Adding in [1, 1, n*h*w, c] shape
        ttnn.deallocate(shortcut)
        out = ttnn.relu(out)
        return out, out_h, out_w


class TtResNet:
    def __init__(self, device, parameters, conv_pt):
        self.stem = TtDeepLabStem(device, parameters.stem, conv_pt.stem)

        res2 = [
            TtBottleneckBlock(device, parameters.res2[0], conv_pt.res2[0]),
            TtBottleneckBlock(device, parameters.res2[1], conv_pt.res2[1]),
            TtBottleneckBlock(device, parameters.res2[2], conv_pt.res2[2]),
        ]

        res3 = [
            TtBottleneckBlock(device, parameters.res3[0], conv_pt.res3[0]),
            TtBottleneckBlock(device, parameters.res3[1], conv_pt.res3[1]),
            TtBottleneckBlock(device, parameters.res3[2], conv_pt.res3[2]),
            TtBottleneckBlock(device, parameters.res3[3], conv_pt.res3[3]),
        ]

        res4 = [
            TtBottleneckBlock(device, parameters.res4[0], conv_pt.res4[0], shard_type="WS"),
            TtBottleneckBlock(device, parameters.res4[1], conv_pt.res4[1]),
            TtBottleneckBlock(device, parameters.res4[2], conv_pt.res4[2]),
            TtBottleneckBlock(device, parameters.res4[3], conv_pt.res4[3]),
            TtBottleneckBlock(device, parameters.res4[4], conv_pt.res4[4]),
            TtBottleneckBlock(device, parameters.res4[5], conv_pt.res4[5]),
        ]

        res5 = [
            TtBottleneckBlock(device, parameters.res5[0], conv_pt.res5[0], shard_type=None),
            TtBottleneckBlock(device, parameters.res5[1], conv_pt.res5[1], shard_type=None),
            TtBottleneckBlock(device, parameters.res5[2], conv_pt.res5[2], shard_type=None),
        ]

        self.stages = [res2, res3, res4, res5]

        self._out_features = ["res2", "res3", "res5"]
        self.stage_names = ("res2", "res3", "res4", "res5")

        self.size_divisibility = 0

    def run_stage(self, x, out_h, out_w, stage):
        for block in stage:
            x, out_h, out_w = block(x, out_h, out_w)
        return x, out_h, out_w

    def __call__(self, x, inp_h, inp_w):
        if use_signpost:
            signpost(header="ResNet")
        outputs = {}
        x, out_h, out_w = self.stem(x, inp_h, inp_w)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        for name, stage in zip(self.stage_names, self.stages):
            x, out_h, out_w = self.run_stage(x, out_h, out_w, stage)
            if name in self._out_features:
                x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
                outputs[name] = (x, out_h, out_w)
        return outputs


class TtPanopticDeepLab:
    def __init__(self, device, parameters, conv_pt):
        self.device = device
        self.backbone = TtResNet(device, parameters.backbone, conv_pt.backbone)
        self.sem_seg_head = TtPanopticDeepLabSemSegHead(device, parameters.sem_seg_head, conv_pt.sem_seg_head)
        self.ins_embed_head = TtPanopticDeepLabInsEmbedHead(device, parameters.ins_embed_head, conv_pt.ins_embed_head)

        self.input_tensor = None
        self.output0 = None
        self.output1 = None
        self.output2 = None

    def __call__(self, x=None, inp_h=256, inp_w=512):
        if use_signpost:
            signpost(header="PanopticDeepLab")

        # print(x)
        # print(self.input_tensor)
        x = x if x is not None else self.input_tensor
        # if x is not None:
        #     x = self.input_tensor
        #     ttnn.reallocate(x)
        # print("Input is : ")
        # print(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        features = self.backbone(x, inp_h, inp_w)  # Resnet frontend

        sem_seg_results = self.sem_seg_head(features)[0]  #
        sem_seg_results = ttnn.permute(sem_seg_results, (0, 3, 1, 2))  # (TTNN) NHWC -> (TORCH) NCHW shape

        center_results, offset_results = self.ins_embed_head(features)
        center_results = ttnn.permute(center_results, (0, 3, 1, 2))  # (TTNN) NHWC -> (TORCH) NCHW shape
        offset_results = ttnn.permute(offset_results, (0, 3, 1, 2))  # (TTNN) NHWC -> (TORCH) NCHW shape

        self.output0 = sem_seg_results
        self.output1 = center_results
        self.output2 = offset_results

        outputs = {
            "sem_seg_results": sem_seg_results,
            "center_results": center_results,
            "offset_results": offset_results,
        }

        return outputs
