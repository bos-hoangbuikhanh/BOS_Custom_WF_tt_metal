import copy
import warnings

import numpy as np
import torch
import ttnn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence

from bos_metal import device_box
from tt.projects.configs.ops_config import MyDict

from ..utils import pt2tt
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer


@TRANSFORMER_LAYER_SEQUENCE.register_module(name="BEVFormerEncoder_tt")
class BEVFormerEncoder(TransformerLayerSequence):
    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(
        self,
        *args,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        dataset_type="nuscenes",
        bev_h=100,
        bev_w=100,
        **kwargs,
    ):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

        self.max_len = [3_072, 1_376, 1_376, 3_680, 992, 992]
        # [3_072, 1_344, 1_376, 3_680, 928, 992]

        self.ref_2d = pt2tt(
            self.get_reference_points(bev_h, bev_w, dim="2d"),
            device=device_box.get(),
            dtype=ttnn.bfloat16,
        )
        self.ref_3d_denormalizer = [pt2tt(torch.tensor([20.0, 12.0]).reshape(1, 1, 2).repeat(1, self.max_len[i], 4), device=device_box.get()) for i in range(6)]
        self.reference_points_rebatch_zeros = [ttnn.zeros(shape=(1, self.max_len[i], 8), dtype=ttnn.bfloat16, device=device_box.get(), layout=ttnn.TILE_LAYOUT) for i in range(6)]
        weight_hash_config_case = ttnn.BilinearWeightHashConfig(
            step_x=100,
            step_y=100,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.bilinear_weight_hash = ttnn.bos_create_bilinear_hash(device_box.get(), **weight_hash_config_case)

    @staticmethod
    def get_reference_points(
        H,
        W,
        Z=8,
        num_points_in_pillar=4,
        dim="3d",
        bs=1,
        device=None,
        dtype=torch.float,
    ):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == "3d":
            zs = (
                torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device)
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def point_sampling(self, img_metas, device=None, debug=False):
        lidar2img = ttnn.stack(img_metas[0]["lidar2img"], 0)
        lidar2img = ttnn.experimental.view(lidar2img, (1, 1, 6, 4, 4))
        lidar2img = ttnn.repeat(lidar2img, ttnn.Shape([4, 1, 1, 1, 1]))

        reference_points_cam = ttnn.matmul(
            self.reference_points,
            lidar2img,
            transpose_b=True,
        )

        reference_points_cam = ttnn.permute(reference_points_cam, (0, 4, 1, 2, 3))
        ref_z = reference_points_cam[:, 2:3]

        eps = 1e-5
        bev_mask = ttnn.gt(ref_z, eps, memory_config=ttnn.L1_MEMORY_CONFIG)
        # broad-cast is potentially erroneous -> manually repeat
        tmp = ttnn.maximum(ref_z, ttnn.full_like(ref_z, eps))
        tmp = ttnn.repeat_interleave(tmp, 2, 1)
        reference_points_cam = ttnn.divide(reference_points_cam[:, 0:2], tmp, memory_config=ttnn.L1_MEMORY_CONFIG)

        ref_x = ttnn.div(reference_points_cam[:, 0:1], 640)  # img_metas[0]['img_shape'][0][1])
        ref_y = ttnn.div(reference_points_cam[:, 1:2], 384)  # img_metas[0]['img_shape'][0][0])

        bev_mask = ttnn.logical_and_(bev_mask, ttnn.gt(ref_x, 0.0))
        bev_mask = ttnn.logical_and_(bev_mask, ttnn.lt(ref_x, 1.0))
        bev_mask = ttnn.logical_and_(bev_mask, ttnn.gt(ref_y, 0.0))
        bev_mask = ttnn.logical_and_(bev_mask, ttnn.lt(ref_y, 1.0))

        # bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = ttnn.concat([ref_x, ref_y], 1)
        # reference_points_cam = ttnn.to_layout(reference_points_cam, ttnn.ROW_MAJOR_LAYOUT)
        reference_points_cam = ttnn.reshape(reference_points_cam, (8, 6, 10_000))
        reference_points_cam = ttnn.permute(reference_points_cam, (1, 2, 0))
        # reference_points_cam = ttnn.repeat(reference_points_cam, (1, 1, 16))

        bev_mask = ttnn.squeeze(ttnn.permute(bev_mask, (1, 3, 2, 4, 0)), 0)

        return reference_points_cam, bev_mask

    def forward(
        self,
        bev_query,
        key,
        value,
        *args,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        shift=0.0,
        memory_config=MyDict(),
        program_config=MyDict(),
        debug=False,
        **kwargs,
    ):
        bev_query = ttnn.to_memory_config(bev_query, memory_config["bev_query"].value)
        bev_query = ttnn.reallocate(bev_query)

        intermediate = []

        reference_points_cam, bev_mask = self.point_sampling(
            kwargs["img_metas"],
            device=bev_query.device(),
            debug=debug,
        )
        shift_ref_2d = self.ref_2d + shift
        ttnn.deallocate(shift)

        bs, len_bev, num_bev_level, _ = self.ref_2d.shape
        if prev_bev is not None:
            # [bs * 2, len_bev, -1]
            prev_bev = ttnn.concat(
                [prev_bev, ttnn.sharded_to_interleaved(bev_query, memory_config=ttnn.L1_MEMORY_CONFIG)], 0
            )
            # [bs * 2, len_bev, num_bev_level, 2]
            hybird_ref_2d = ttnn.concat([shift_ref_2d, self.ref_2d], 0)
            ttnn.deallocate(shift_ref_2d)
        else:
            hybird_ref_2d = ttnn.concat([self.ref_2d, self.ref_2d], 0)
        hybird_ref_2d = ttnn.multiply_(hybird_ref_2d, 100.0)
        hybird_ref_2d = ttnn.sub_(hybird_ref_2d, 0.5)
        hybird_ref_2d = ttnn.to_layout(hybird_ref_2d, ttnn.ROW_MAJOR_LAYOUT)
        hybird_ref_2d = ttnn.permute(hybird_ref_2d, (2, 1, 0, 3)) # 1, 10_000, 2, 2
        hybird_ref_2d = ttnn.repeat(hybird_ref_2d, (1, 1, 1, 4)) # 1, 10_000, 2, 8
        hybird_ref_2d = ttnn.reshape(hybird_ref_2d, (1, 100 * 100, 16))
        hybird_ref_2d = ttnn.repeat(hybird_ref_2d, (1, 1, 8)) # 1, 10_000, 128
        # hybird_ref_2d = ttnn.to_layout(hybird_ref_2d, ttnn.TILE_LAYOUT)

        indexes = []
        bev_mask_sums = ttnn.sum(bev_mask, -1)
        ttnn.deallocate(bev_mask)

        # TODO: Edit ttnn.nonzero to take in 6xN array instead of looping 6 times
        groups = {(1, 2): [], (4, 5): []}
        
        reference_points_rebatch_lst = []

        for i in range(bev_mask.shape[0]):
            reference_points_rebatch = ttnn.clone(self.reference_points_rebatch_zeros[i])
            _, indices = ttnn.bos_nonzero(
                ttnn.to_layout(bev_mask_sums[i], ttnn.ROW_MAJOR_LAYOUT),
                max_length=self.max_len[i],
            )
            reference_points_rebatch = ttnn.operations.moreh.getitem(reference_points_cam[i], [indices], [0], memory_config=ttnn.L1_MEMORY_CONFIG)
            indexes.append(indices)

            reference_points_rebatch = ttnn.multiply_(reference_points_rebatch, self.ref_3d_denormalizer[i])
            reference_points_rebatch = ttnn.sub_(reference_points_rebatch, 0.5)
            reference_points_rebatch = ttnn.repeat(reference_points_rebatch, (1, 1, 16), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            reference_points_rebatch_lst.append(reference_points_rebatch)

            for idx in groups.keys():
                if i in idx:
                    groups[idx].append(reference_points_rebatch)

        for idx, tensors in groups.items():
            if len(tensors) > 0:
                concat_tensor = ttnn.concat(tensors, dim=0)
                [ttnn.deallocate(tensor) for tensor in tensors]
                for id in idx:
                    ttnn.deallocate(reference_points_rebatch_lst[id])
                    reference_points_rebatch_lst[id] = concat_tensor
        ttnn.deallocate(reference_points_cam)
        ttnn.deallocate(bev_mask)

        count = ttnn.gt(bev_mask_sums, 0.0)
        ttnn.deallocate(bev_mask_sums)
        count = ttnn.sum(count, 0)
        count = ttnn.unsqueeze(ttnn.clamp(count, min=1.0, output_tensor=count), -1)
        count = ttnn.reallocate(count)

        bs, num_cams, l, embed_dims = key.shape
        value = ttnn.reshape(value, (bs * num_cams, l, self.embed_dims))
        groups = [[0], [1, 2], [3], [4, 5]]
        value_spatial = [value[group[0]:group[-1]+1] for group in groups]
        ttnn.deallocate(value)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value_spatial,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_rebatch=reference_points_rebatch_lst,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                memory_config=memory_config,
                program_config=program_config,
                indexes=indexes,
                count=count,
                bilinear_weight_hash=self.bilinear_weight_hash,
                **kwargs,
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return ttnn.concat([ttnn.unsqueeze(inter, 0) for inter in intermediate], 0)

        return output


@TRANSFORMER_LAYER.register_module(name="BEVFormerLayer_tt")
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    counter = 0
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
        self,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

        self.temporal_spatial_shapes = ttnn.Tensor(
            data=[100, 100],
            data_type=ttnn.bfloat16,
            shape=[1, 1, 1, 2],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device_box.get(),
        ).reshape([1, 2])

    def forward(
        self,
        query,
        key=None,
        value=None,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        reference_points_cam=None,
        reference_points_rebatch=None,
        mask=None,
        spatial_shapes=None,
        prev_bev=None,
        bilinear_weight_hash=None,
        memory_config=MyDict(),
        program_config=MyDict(),
        indexes=None,
        count=None,
        value_spatial=None,
        **kwargs,
    ):

        norm_index, attn_index, ffn_index = 0, 0, 0
        identity = query

        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            # temporal self attention
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=self.temporal_spatial_shapes,
                    bilinear_weight_hash=bilinear_weight_hash,
                    memory_config=memory_config["self_attn"],
                    program_config=program_config["self_attn"],
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = ttnn.sharded_to_interleaved(query, memory_config=ttnn.L1_MEMORY_CONFIG)
                query = self.norms[norm_index](
                    query,
                    memory_config=memory_config["norm"].value,
                    program_config=program_config["norm"].value,
                )
                query = ttnn.to_memory_config(query, identity.memory_config())
                ttnn.deallocate(identity)
                query = ttnn.reallocate(query)
                norm_index += 1

            # spatial cross attention
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points_rebatch=reference_points_rebatch,
                    spatial_shapes=spatial_shapes,
                    indexes=indexes,
                    count=count,
                    bilinear_weight_hash=bilinear_weight_hash,
                    memory_config=memory_config["cross_attn"],
                    program_config=program_config["cross_attn"],
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](
                    query,
                    identity if self.pre_norm else None,
                    memory_config=memory_config["ffn"],
                    program_config=program_config["ffn"],
                )
                ffn_index += 1

        BEVFormerLayer.counter += 1
        return query
