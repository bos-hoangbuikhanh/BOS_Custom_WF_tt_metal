import ttnn
from models.bos_model.oft.ttnn.ttnn_oft import TtOft
from models.bos_model.oft.ttnn.ttnn_resnet import TtBasicBlock, TtConv, TtGroupNorm, TtResNetFeatures

try:
    from tracy import signpost

    use_signpost = True

except ModuleNotFoundError:
    use_signpost = False

SliceHeight = ttnn.Conv2dSliceHeight
SliceWidth = ttnn.Conv2dSliceWidth


class TtOftNet:
    def __init__(self, device, parameters, conv_pt, block, layers, topdown_layers=8, grid_res=0.5, grid_height=4.0):
        self.frontend = TtResNetFeatures(
            device, parameters.frontend, conv_pt.frontend, block, layers, layer_name="frontend"
        )

        self.lat8 = TtConv(parameters.lat8, conv_pt.lat8, stride=1, padding=0)
        self.lat16 = TtConv(parameters.lat16, conv_pt.lat16, stride=1, padding=0)
        self.lat32 = TtConv(parameters.lat32, conv_pt.lat32, stride=1, padding=0)

        self.gn8 = TtGroupNorm(
            device,
            parameters.bn8,
            num_channels=conv_pt.lat8.out_channels,
            fallback_on_groupnorm=False,
            layer_name=f"bn8",
        )

        self.gn16 = TtGroupNorm(
            device,
            parameters.bn16,
            num_channels=conv_pt.lat16.out_channels,
            fallback_on_groupnorm=False,
            layer_name=f"bn16",
        )

        self.gn32 = TtGroupNorm(
            device,
            parameters.bn32,
            num_channels=conv_pt.lat32.out_channels,
            fallback_on_groupnorm=False,
            layer_name=f"bn32",
        )

        self.oft8 = TtOft(device, parameters.oft8, grid_res, grid_height, 1 / 8.0)
        self.oft16 = TtOft(device, parameters.oft16, grid_res, grid_height, 1 / 16.0)
        self.oft32 = TtOft(device, parameters.oft32, grid_res, grid_height, 1 / 32.0)

        TtBasicBlock.id = 1
        self.topdown = [
            TtBasicBlock(
                device,
                parameters.topdown[i],
                conv_pt.topdown[i],
                256,
                256,
                stride=1,
                height_sharding=False,
                slice_type=SliceHeight,
                num_slices=4,
                layer="topdown",
                layer_name=f"topdown.{i}",
            )
            for i in range(topdown_layers)
        ]

        self.head = TtConv(parameters.head, conv_pt.head, stride=1, padding=1, slice_type=SliceHeight, num_slices=4)

    def __call__(self, device, image, pre_config=None):
        if use_signpost:
            signpost(header="OftNet")

        feats8, feats16, feats32 = self.frontend(device, image)
        ttnn.deallocate(image)
        lat8, out_h, out_w = self.lat8(device, feats8)
        ttnn.deallocate(feats8)
        if lat8.is_sharded():
            lat8 = ttnn.sharded_to_interleaved(lat8, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)

        lat8 = self.gn8(lat8, out_h, out_w)
        lat8 = ttnn.reshape(
            lat8,
            (lat8.shape[0], out_h, out_w, lat8.shape[-1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        lat8 = ttnn.relu(lat8, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lat8 = ttnn.permute(lat8, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lat8 = ttnn.typecast(lat8, ttnn.float32)

        lat16, out_h, out_w = self.lat16(device, feats16)
        ttnn.deallocate(feats16)

        if lat16.is_sharded():
            lat16 = ttnn.sharded_to_interleaved(lat16, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)

        lat16 = self.gn16(lat16, out_h, out_w)
        lat16 = ttnn.reshape(
            lat16,
            (lat16.shape[0], out_h, out_w, lat16.shape[-1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        lat16 = ttnn.relu(lat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lat16 = ttnn.permute(lat16, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lat16 = ttnn.typecast(lat16, ttnn.float32)

        lat32, out_h, out_w = self.lat32(device, feats32)
        ttnn.deallocate(feats32)
        if lat32.is_sharded():
            lat32 = ttnn.sharded_to_interleaved(lat32, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)

        lat32 = self.gn32(lat32, out_h, out_w)
        lat32 = ttnn.reshape(
            lat32,
            (lat32.shape[0], out_h, out_w, lat32.shape[-1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        lat32 = ttnn.relu(lat32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lat32 = ttnn.permute(lat32, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lat32 = ttnn.typecast(lat32, ttnn.float32)

        ortho8 = self.oft8(lat8, pre_config["oft8"])
        ttnn.deallocate(lat8)
        ortho16 = self.oft16(lat16, pre_config["oft16"])
        ttnn.deallocate(lat16)
        ortho32 = self.oft32(lat32, pre_config["oft32"])
        ttnn.deallocate(lat32)

        ortho = ttnn.add(ortho8, ortho16, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(ortho8)
        ttnn.deallocate(ortho16)

        topdown = ttnn.add(ortho, ortho32, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(ortho)
        ttnn.deallocate(ortho32)

        topdown = ttnn.permute(topdown, [0, 2, 3, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        for layer in self.topdown:
            topdown = layer(device, topdown)

        batch, depth, width, _ = topdown.shape
        outputs, out_h, out_w = self.head(device, topdown)
        ttnn.deallocate(topdown)

        if outputs.is_sharded():
            outputs = ttnn.sharded_to_interleaved(outputs, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)

        outputs = ttnn.permute(outputs, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        outputs = ttnn.reshape(outputs, (batch, -1, 9, depth, width), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        slices = [1, 3, 3, 2]
        start = 0
        parts = []

        for s in slices:
            parts.append(outputs[:, :, start : start + s, :, :])
            start += s
        parts[0] = ttnn.squeeze(parts[0], 2)
        return parts
