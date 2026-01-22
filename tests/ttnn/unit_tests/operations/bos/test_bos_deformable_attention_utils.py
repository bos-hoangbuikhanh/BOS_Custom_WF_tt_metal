from typing import List
import torch
import ttnn
import math
from torch.nn import functional as F


def pt2tt(tensor, device=None, dtype=ttnn.bfloat16, *args, **kwargs):
    """Convert a PyTorch tensor to a TT tensor."""
    if isinstance(tensor, ttnn.Tensor) or tensor is None:
        return tensor
    return ttnn.from_torch(tensor, device=device, dtype=dtype, *args, **kwargs)


def tt2pt(tensor, dtype=torch.float32, *args, **kwargs):
    """Convert a PyTorch tensor to a TT tensor."""
    if isinstance(tensor, torch.Tensor) or tensor is None:
        return tensor
    return ttnn.to_torch(tensor, dtype=dtype, *args, **kwargs)


def multi_scale_deformable_attn_pytorch_ref(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs, num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    # _save_msda(sampling_grids, f"sampling_grids")

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # _save_msda(value_l_, f"value_l_.flatten.{level}")
        # _save_msda(sampling_grid_l_, f"sampling_grid_l.flatten.{level}")
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        # _save_msda(sampling_value_l_, f"sampling_value_l_.grid_sample.{level}")

        sampling_value_list.append(sampling_value_l_)

    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2)
    attention_weights = attention_weights.reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    # _save_msda(attention_weights, f"attention_weights")
    sampling_values = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    # sampling_values.shape
    # torch.Size([16, 32, 40000, 4])
    # attention_weights.shape
    # torch.Size([16, 1, 40000, 4])
    output = (sampling_values * attention_weights).sum(-1)
    # _save_msda(output, f"output.sum")
    output = output.view(bs, num_heads * embed_dims, num_queries)
    # _save_msda(output, f"output.view")
    output = output.transpose(1, 2).contiguous()

    return output


def split_value(value: ttnn.Tensor, spatial_shapes: torch.Tensor) -> List[ttnn.Tensor]:
    value_list = []
    for i, (H_, W_) in enumerate(spatial_shapes):
        if i == 0:
            start = 0
            end = H_ * W_
        else:
            start = end
            end = start + H_ * W_
        value_list.append(value[:, start:end, :, :])
    return value_list


def multi_scale_deformable_attn_ttnn(
    value: ttnn.Tensor,
    value_spatial_shapes: ttnn.Tensor,
    sampling_locations: ttnn.Tensor,
    attention_weights: ttnn.Tensor,
) -> ttnn.Tensor:
    """TTNN version of multi-scale deformable attention.
    Args:
        value (ttnn.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (ttnn.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (ttnn.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (ttnn.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        ttnn.Tensor: has shape (bs, num_queries, embed_dims)
    """
    # value_spatial_shapes = tt2pt(value_spatial_shapes, dtype=torch.int32)
    device = value.device()
    bs, num_keys, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = split_value(value, value_spatial_shapes)
    # sampling_grids = (- 1) + 2 * sampling_locations.tile()
    sampling_grids = sampling_locations * 2 - 1
    # _assert_msda(sampling_grids, f"sampling_grids")

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level]
        value_l_ = ttnn.reshape(value_l_, [bs, num_keys, num_heads * embed_dims])  # flatten(2)
        value_l_ = ttnn.transpose(value_l_, 1, 2)
        value_l_ = ttnn.reshape(value_l_, [bs * num_heads, embed_dims, H_, W_])
        sampling_grid_l_ = sampling_grids[:, :, :, level]
        sampling_grid_l_ = ttnn.transpose(sampling_grid_l_, 1, 2)
        sampling_grid_l_ = ttnn.reshape(sampling_grid_l_, [bs * num_heads, num_queries, num_points, 2])
        value_l_ = ttnn.permute(value_l_, (0, 2, 3, 1))
        value_l_ = ttnn.to_layout(value_l_, ttnn.ROW_MAJOR_LAYOUT)
        sampling_grid_l_ = ttnn.to_layout(sampling_grid_l_, ttnn.ROW_MAJOR_LAYOUT)
        # print("Grid sample")
        # print(f"Value shape: {value_l_.shape}, sampling_grid_l_ shape: {sampling_grid_l_.shape}")
        sampling_value_l_ = ttnn.grid_sample(value_l_, sampling_grid_l_)
        sampling_value_l_ = ttnn.permute(sampling_value_l_, (0, 3, 1, 2))
        sampling_value_list.append(sampling_value_l_)

    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    num_levels = value_spatial_shapes.shape[0]
    attention_weights = attention_weights[:, :, :, :num_levels, :]
    attention_weights = ttnn.transpose(attention_weights, 1, 2)
    attention_weights = ttnn.reshape(attention_weights, [bs * num_heads, 1, num_queries, num_levels * num_points])

    sampling_values = ttnn.stack(sampling_value_list, dim=-2)
    sampling_values = ttnn.reshape(
        sampling_values, [bs * num_heads, embed_dims, num_queries, num_levels * num_points]
    )  # flatten(-1, -2)
    sampling_values = ttnn.to_layout(sampling_values, ttnn.TILE_LAYOUT)
    attention_weights = ttnn.to_layout(attention_weights, ttnn.TILE_LAYOUT)

    output = ttnn.mul_(sampling_values, attention_weights)
    output = ttnn.sum(output, -1)
    output = ttnn.reshape(output, [bs, num_heads * embed_dims, num_queries])
    output = ttnn.transpose(output, 1, 2)

    return output


def generate_parametric_inputs(
    batch_size,
    num_heads,
    num_queries,
    num_levels,
    num_points,
    num_keys,
    embed_dims=4,
    seed=0,
    ratio=0.01,
    **kwargs,
):
    torch.manual_seed(seed)

    # Create spatial_shapes: even split of keys across levels
    keys_per_level = [num_keys // num_levels] * num_levels
    keys_per_level[0] += num_keys % num_levels  # Handle remainder

    # Choose reasonable H × W per level (make H = ceil(sqrt(keys)), W = ceil(keys / H))
    spatial_shapes = []
    for keys in keys_per_level:
        H = int(math.ceil(keys**0.5))
        W = int(math.ceil(keys / H))
        spatial_shapes.append([H, W])
    spatial_shapes_tensor = torch.tensor(spatial_shapes, dtype=torch.int32)

    # Total num_keys may be slightly more than requested due to H×W rounding
    total_keys = sum(H * W for H, W in spatial_shapes)

    # Value: [B, total_keys, num_heads, embed_dims]
    value_size = batch_size * total_keys * num_heads * embed_dims
    value = torch.arange(value_size, dtype=torch.float32)
    value = value.reshape(batch_size, total_keys, num_heads, embed_dims)
    value = value.to(dtype=torch.bfloat16).float() * ratio

    # Sampling locations: [B, Q, H, L, P, 2] (coords normalized in [0, 1])
    sampling_locations = torch.rand(
        batch_size, num_queries, num_heads, num_levels, num_points, 2, dtype=torch.bfloat16
    ).float()

    # Attention weights: [B, Q, H, L, P]
    attention_weights = torch.rand(
        batch_size, num_queries, num_heads, num_levels, num_points, dtype=torch.bfloat16
    ).float()

    return {
        "value": value,
        "value_spatial_shapes": spatial_shapes_tensor,
        "sampling_locations": sampling_locations,
        "attention_weights": attention_weights,
    }


def make_test_case(
    batch_size, num_heads, num_queries, num_levels, num_points, num_keys, embed_dims, seed=0, pcc=0.998, **kwargs
):
    return dict(
        batch_size=batch_size,
        num_heads=num_heads,
        num_queries=num_queries,
        num_levels=num_levels,
        num_points=num_points,
        num_keys=num_keys,
        embed_dims=embed_dims,
        seed=seed,
        pcc=pcc,
        **kwargs,
    )


def make_weight_hash_config(
    step_x, step_y, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
):
    return dict(
        step_x=step_x,
        step_y=step_y,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )
