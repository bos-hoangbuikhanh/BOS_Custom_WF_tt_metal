import torch.nn.functional as F

import ttnn
from models.bos_model.oft.ttnn.model_preprocessing import _nearest_32_per_core, gn_configs

try:
    from tracy import signpost

    use_signpost = True

except ModuleNotFoundError:
    use_signpost = False


class TtConv:
    def __init__(
        self,
        parameters,
        conv_pt,
        *,
        stride=1,
        padding=1,
        has_bias=False,
        act_block_h=32,
        reshard=False,
        deallocate=False,
        height_sharding=False,
        activation="",
        width_sharding=False,
        block_sharding=False,
        dtype=ttnn.bfloat16,
        slice_type=None,
        num_slices=None,
        fp32_dest_acc_en=True,
    ) -> None:
        self.weights = parameters.weight

        self.conv_pt = conv_pt
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = parameters.bias

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.stride = stride
        self.padding = padding
        self.out_channels = conv_pt.out_channels
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.dtype = dtype
        self.slice_type = slice_type
        self.num_slices = num_slices
        self.fp32_dest_acc_en = fp32_dest_acc_en

        if width_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        elif height_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif block_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        else:
            self.shard_layout = None

        self.deallocate = deallocate
        self.activation = activation

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape if self.has_bias else ''} {self.kernel_size}"

    def __call__(self, device, input_tensor):
        # if use_signpost:
        #     signpost(header="Conv2d")
        if self.slice_type is not None and self.num_slices is not None:
            slice_config = ttnn.Conv2dSliceConfig(
                slice_type=self.slice_type,
                num_slices=self.num_slices,
            )
        else:
            slice_config = None
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.dtype,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate,
            activation=self.activation,
            reshard_if_not_optimal=self.reshard,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [output_tensor, [out_h, out_w], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias if self.has_bias else None,
            in_channels=self.conv_pt.in_channels,
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            batch_size=self.conv_pt.batch_size,
            input_height=self.conv_pt.input_height,
            input_width=self.conv_pt.input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            slice_config=slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return output_tensor, out_h, out_w


class TtGroupNorm:
    def __init__(
        self,
        device,
        parameters,
        num_channels,
        num_groups=16,
        grid_shape=(1, 4),
        num_out_blocks=2,
        eps=1e-05,
        inplace=False,
        fallback_on_groupnorm=False,
        layer_name="",
    ):
        self.device = device
        self.eps = eps
        self.batch_size = 1
        self.C = num_channels
        self.num_groups = num_groups
        self.inplace = inplace
        # self.grid_size = ttnn.CoreGrid(y=grid_shape[1], x=grid_shape[0])
        self.grid_size = gn_configs[layer_name]["grid_size"]
        self.num_out_blocks = gn_configs[layer_name]["num_out_blocks"]

        self.fallback_on_gn_weight = parameters.weight
        self.fallback_on_gn_bias = parameters.bias

        self.fallback_on_groupnorm = fallback_on_groupnorm

        weight = ttnn.create_group_norm_weight_bias_rm(parameters.weight, self.C, self.grid_size.y)
        self.weight = ttnn.from_torch(
            weight,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        bias = ttnn.create_group_norm_weight_bias_rm(parameters.bias, self.C, self.grid_size.y)
        self.bias = ttnn.from_torch(
            bias,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        input_mask_tensor = ttnn.create_group_norm_input_mask(self.C, self.num_groups, self.grid_size.y)
        self.input_mask_tensor = ttnn.from_torch(
            input_mask_tensor,
            dtype=ttnn.DataType.BFLOAT8_B,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x, out_h, out_w):
        if self.fallback_on_groupnorm:
            if use_signpost:
                signpost(header="TORCH GroupNorm")

            x = ttnn.to_torch(x)
            x = F.group_norm(
                x.reshape(x.shape[0], out_h, out_w, x.shape[-1]).permute((0, 3, 1, 2)),
                num_groups=self.num_groups,
                weight=self.fallback_on_gn_weight,
                bias=self.fallback_on_gn_bias,
                eps=self.eps,
            )

            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            x = ttnn.permute(x, (0, 2, 3, 1))

        else:
            # if use_signpost:
            #     signpost(header="TTNN GroupNorm")

            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

            unpadded_shape = x.shape
            out_shape = [
                unpadded_shape[0],
                unpadded_shape[1],
                _nearest_32_per_core(unpadded_shape[2], self.grid_size.x),
                _nearest_32_per_core(unpadded_shape[3], self.grid_size.y),
            ]

            x = ttnn.tilize_with_val_padding(x, output_tensor_shape=out_shape, pad_value=0, use_multicore=True)
            x = ttnn.group_norm(
                x,
                num_groups=self.num_groups,
                epsilon=self.eps,
                input_mask=self.input_mask_tensor,
                weight=self.weight,
                bias=self.bias,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                core_grid=self.grid_size,
                inplace=self.inplace,
                num_out_blocks=self.num_out_blocks,
                output_layout=ttnn.TILE_LAYOUT,
            )
            x = x[:, :, : out_h * out_w, : x.shape[-1]]  # unpad

        return x


class TtBasicBlock:
    expansion = 1
    id = 1

    def __init__(
        self,
        device,
        parameters,
        conv_pt,
        inplanes,
        planes,
        stride=1,
        height_sharding=False,
        act_block_h=32,
        layer=None,
        slice_type=None,
        num_slices=None,
        layer_name="",
    ):
        self.inplanes = inplanes
        self.planes = planes
        self.slice_type = slice_type
        self.num_slices = num_slices
        self.layer = layer
        self.id = TtBasicBlock.id
        TtBasicBlock.id += 1

        self.conv1 = TtConv(
            parameters.conv1,
            conv_pt.conv1,
            stride=stride,
            height_sharding=height_sharding,
            act_block_h=act_block_h,
            slice_type=slice_type,
            num_slices=num_slices,
        )

        self.gn1 = TtGroupNorm(
            device,
            parameters.bn1,
            num_channels=conv_pt.conv1.out_channels,
            fallback_on_groupnorm=False,
            layer_name=f"{layer_name}.bn1",
        )

        self.conv2 = TtConv(
            parameters.conv2,
            conv_pt.conv2,
            height_sharding=height_sharding,
            act_block_h=act_block_h,
            slice_type=slice_type,
            num_slices=num_slices,
        )

        self.gn2 = TtGroupNorm(
            device,
            parameters.bn2,
            num_channels=conv_pt.conv2.out_channels,
            fallback_on_groupnorm=False,
            layer_name=f"{layer_name}.bn2",
        )

        self.layer = layer
        if stride != 1 or inplanes != planes:
            self.downsample = True

            self.downsample_conv = TtConv(
                parameters.downsample[0],
                conv_pt.downsample[0],
                stride=stride,
                padding=0,
                height_sharding=height_sharding,
                act_block_h=act_block_h,
            )

            self.downsample_gn = TtGroupNorm(
                device,
                parameters.downsample[1],
                num_channels=conv_pt.downsample[0].out_channels,
                fallback_on_groupnorm=False,
                layer_name=f"{layer_name}.downsample.1",
            )
        else:
            self.downsample = None

    def __call__(self, device, x):
        if use_signpost:
            header = f"BasicBlock-{self.layer}-{self.id}" if self.layer else f"BasicBlock-{self.id}"
            signpost(header=header)
        out, out_h, out_w = self.conv1(device, x)
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)

        out = self.gn1(out, out_h, out_w)
        out = ttnn.relu(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self.layer == "topdown":
            out = ttnn.to_memory_config(
                out, ttnn.DRAM_MEMORY_CONFIG
            )  # Conv DRAM expects the input tensor to be in DRAM.

        out, out_h, out_w = self.conv2(device, out)
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)

        out = self.gn2(out, out_h, out_w)
        out = ttnn.reshape(out, (1, out_h, out_w, out.shape[-1]))

        if self.downsample is not None:
            x, out_h, out_w = self.downsample_conv(device, x)
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

            x = self.downsample_gn(x, out_h, out_w)
            x = ttnn.reshape(x, (1, out_h, out_w, x.shape[-1]))

        if out.is_sharded():
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)

        if x.get_layout() == ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        out += x
        ttnn.deallocate(x)
        out = ttnn.relu(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self.layer == "topdown":
            out = ttnn.to_memory_config(
                out, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )  # Conv DRAM expects the input tensor to be in DRAM.

        return out


class TtResNetFeatures:
    def __init__(self, device, parameters, conv_pt, block, layers, layer_name=""):
        self.inplanes = 64

        self.conv1 = TtConv(
            parameters.conv1, conv_pt.conv1, stride=2, padding=3, fp32_dest_acc_en=False, height_sharding=False
        )

        self.gn1 = TtGroupNorm(
            device,
            parameters.bn1,
            num_channels=conv_pt.conv1.out_channels,
            fallback_on_groupnorm=False,
            layer_name=f"{layer_name}.bn1",
        )

        self.layer1 = self._make_layer(
            device, parameters.layer1, conv_pt.layer1, block, 64, layers[0], layer_name=f"{layer_name}.layer1"
        )
        self.layer2 = self._make_layer(
            device,
            parameters.layer2,
            conv_pt.layer2,
            block,
            128,
            layers[1],
            stride=2,
            layer_name=f"{layer_name}.layer2",
        )
        self.layer3 = self._make_layer(
            device,
            parameters.layer3,
            conv_pt.layer3,
            block,
            256,
            layers[2],
            stride=2,
            height_sharding=False,
            layer_name=f"{layer_name}.layer3",
        )
        self.layer4 = self._make_layer(
            device,
            parameters.layer4,
            conv_pt.layer4,
            block,
            512,
            layers[3],
            stride=2,
            height_sharding=False,
            layer_name=f"{layer_name}.layer4",
        )

    def _make_layer(
        self, device, parameters, conv_pt, block, planes, blocks, stride=1, height_sharding=False, layer_name=""
    ):
        layers = []
        layers.append(
            block(
                device,
                parameters[0],
                conv_pt[0],
                self.inplanes,
                planes,
                stride,
                height_sharding=height_sharding,
                layer_name=f"{layer_name}.0",
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    device,
                    parameters[i],
                    conv_pt[i],
                    self.inplanes,
                    planes,
                    height_sharding=height_sharding,
                    layer_name=f"{layer_name}.{i}",
                )
            )

        return layers

    def _run_layer(self, device, x, layer):
        for block in layer:
            x = block(device, x)
        return x

    def __call__(self, device, x):
        if use_signpost:
            signpost(header="ResNetFeatures")

        conv1, out_h, out_w = self.conv1(device, x)
        ttnn.deallocate(x)

        shape = (1, out_h, out_w, conv1.shape[3])
        conv1 = ttnn.sharded_to_interleaved(conv1, ttnn.DRAM_MEMORY_CONFIG)
        conv1 = self.gn1(conv1, out_h, out_w)
        conv1 = ttnn.relu(conv1)

        # shape = (1, 1, conv1.shape[0] * conv1.shape[1] * conv1.shape[2], conv1.shape[3])
        # conv1_in = ttnn.reshape(conv1, shape)
        # ttnn.deallocate(conv1)

        if conv1.get_layout() == ttnn.TILE_LAYOUT:
            conv1 = ttnn.to_layout(
                conv1,
                ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
        conv1 = ttnn.max_pool2d(
            input_tensor=conv1,
            batch_size=conv1.shape[0],
            input_h=shape[1],
            input_w=shape[2],
            channels=conv1.shape[-1],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            # memory_config=ttnn.L1_MEMORY_CONFIG,
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            in_place_halo=True,
        )
        conv1_out_h, conv1_out_w = int((shape[1] + 1) / 2), int((shape[2] + 1) / 2)
        conv1 = ttnn.reshape(
            conv1,
            (shape[0], conv1_out_h, conv1_out_w, shape[3]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if conv1.is_sharded():
            conv1 = ttnn.sharded_to_interleaved(conv1, ttnn.DRAM_MEMORY_CONFIG)

        if conv1.get_layout() != ttnn.TILE_LAYOUT:
            conv1 = ttnn.to_layout(
                conv1,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )

        feats4 = self._run_layer(device, conv1, self.layer1)
        ttnn.deallocate(conv1)
        feats8 = self._run_layer(device, feats4, self.layer2)
        ttnn.deallocate(feats4)
        feats16 = self._run_layer(device, feats8, self.layer3)
        feats32 = self._run_layer(device, feats16, self.layer4)

        return [feats8, feats16, feats32]
