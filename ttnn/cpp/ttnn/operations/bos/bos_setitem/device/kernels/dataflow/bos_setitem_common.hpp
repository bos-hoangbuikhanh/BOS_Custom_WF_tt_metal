// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>
#include <stdint.h>
#include "dataflow_api.h"

#define ALWI inline __attribute__((always_inline))

/**
 * Handles negative indices by wrapping around to the end of the dimension
 * @param index Potentially negative index
 * @param size Size of the dimension
 * @return Wrapped positive index
 */
ALWI uint32_t wrap_index(int32_t index, uint32_t size) { return index < 0 ? size + index : index; }

/**
 * @brief Maps a stick ID from the set/slice space to the corresponding stick ID in the original input tensor.
 *
 * This function calculates the destination stick ID in the input tensor for a given source stick ID from the slice.
 * It handles various slicing methods, including basic start/step indexing and advanced indexing with tensors.
 * The calculation iterates from the inner-most to outer-most dimensions (excluding the last dimension, which is
 * stick-level), converting the multi-dimensional index in the slice space to the corresponding index in the input
 * tensor's space.
 *
 * For example, for a simple slice with a start index of 2 and a step of 1, a `stick_id` of 0 in the slice
 * will be mapped to stick 2 in the input tensor.
 *
 * @param stick_id The ID of the stick within the slice being set.
 * @param input_shape Shape of the original input tensor.
 * @param input_set_shape Shape of the slice being set.
 * @param begins An array of start indices for each dimension of the slice.
 * @param steps An array of step values for each dimension of the slice.
 * @param is_ttnn_tensor A boolean array indicating if a dimension is indexed by a tensor.
 * @param ttnn_tensor An array of pointers to tensors used for advanced indexing.
 * @param n_ttnn_tensor The number of tensors used for advanced indexing.
 * @param tensor_rank The rank of the input tensor.
 * @return The calculated stick ID in the original input tensor.
 */
ALWI uint32_t find_input_stick_id(
    uint32_t stick_id,
    tt_l1_ptr uint32_t* input_shape,
    tt_l1_ptr uint32_t* input_set_shape,
    tt_l1_ptr uint32_t* begins,
    tt_l1_ptr uint32_t* steps,
    tt_l1_ptr uint32_t* is_ttnn_tensor,
    volatile tt_l1_ptr uint32_t** ttnn_tensor,
    uint32_t n_ttnn_tensor,
    uint32_t tensor_rank) {
    int ptr = n_ttnn_tensor - 1;
    int sync_id_tensor_index = -1;
    uint32_t i = 1;
    uint32_t accum_prod_input_shape = 1;
    uint32_t accum_prod_input_set_shape = 1;
    uint32_t res = 0;
    bool flag = false;
    while (i < tensor_rank) {
        i++;
        uint32_t dim_index = tensor_rank - i;
        uint32_t slice_idx = (stick_id / accum_prod_input_set_shape) % input_set_shape[dim_index];
        uint32_t slice_val;
        if (is_ttnn_tensor[dim_index]) {
            sync_id_tensor_index = sync_id_tensor_index == -1 ? slice_idx : sync_id_tensor_index;
            int32_t tmp = (int32_t)ttnn_tensor[ptr][sync_id_tensor_index];
            slice_val = wrap_index(tmp, input_shape[dim_index]);
            ptr--;
        } else {
            slice_val = begins[dim_index] + steps[dim_index] * slice_idx;
        }
        res += slice_val * accum_prod_input_shape;

        if (!is_ttnn_tensor[dim_index] || !flag) {
            accum_prod_input_set_shape *= input_set_shape[dim_index];
            flag = 1;
        }
        accum_prod_input_shape = accum_prod_input_shape * input_shape[dim_index];
    }
    return res;
}

/**
 * @brief Maps a tile ID from the set/slice space to the corresponding tile ID in the original input tensor.
 *
 * This function is analogous to `find_input_stick_id` but operates on tiles instead of sticks. It calculates
 * the destination tile ID in the input tensor for a given source tile ID from the slice. The function accommodates
 * various slicing methods, including basic start/step indexing and advanced indexing with tensors.
 *
 * The calculation iterates from the inner-most to outer-most dimensions, converting the multi-dimensional tile
 * index in the slice space to the corresponding index in the input tensor's space. It specifically handles
 * the tile layout (32x32 blocks) for the last two dimensions by converting their shapes to tile-based shapes.
 * For example, if the `input_set_shape` is `[5, 7, 32, 64]`, it is treated as `[5, 7, 1, 2]` for tile mapping purposes,
 * where the last two dimensions are divided by the tile dimension (32).
 *
 * @param tile_id The ID of the tile within the slice being set.
 * @param input_shape Shape of the original input tensor.
 * @param input_set_shape Shape of the slice being set.
 * @param begins An array of start indices for each dimension of the slice.
 * @param steps An array of step values for each dimension of the slice.
 * @param is_ttnn_tensor A boolean array indicating if a dimension is indexed by a tensor.
 * @param ttnn_tensor An array of pointers to tensors used for advanced indexing.
 * @param n_ttnn_tensor The number of tensors used for advanced indexing.
 * @param tensor_rank The rank of the input tensor.
 * @return The calculated tile ID in the original input tensor.
 */
ALWI uint32_t find_input_tile_id(
    uint32_t tile_id,
    tt_l1_ptr uint32_t* input_shape,
    tt_l1_ptr uint32_t* input_set_shape,
    tt_l1_ptr uint32_t* begins,
    tt_l1_ptr uint32_t* steps,
    tt_l1_ptr uint32_t* is_ttnn_tensor,
    volatile tt_l1_ptr uint32_t** ttnn_tensor,
    uint32_t n_ttnn_tensor,
    uint32_t tensor_rank) {
    int ptr = n_ttnn_tensor - 1;
    int sync_id_tensor_index = -1;
    uint32_t i = 0;
    uint32_t accum_prod_origin_tiles = 1;
    uint32_t accum_prod_set_tiles = 1;
    uint32_t res = 0;
    bool flag = false;
    bool is_last_two_dim = true;

    while (i < tensor_rank) {
        i++;
        if (i > 2) {
            is_last_two_dim = false;
        }
        uint32_t dim_index = tensor_rank - i;

        uint32_t tile_idx = is_last_two_dim
                                ? (tile_id / accum_prod_set_tiles) % ((input_set_shape[dim_index] + 31) / 32)
                                : (tile_id / accum_prod_set_tiles) % input_set_shape[dim_index];

        uint32_t tile_val;
        if (is_ttnn_tensor[dim_index]) {
            sync_id_tensor_index = sync_id_tensor_index == -1 ? tile_idx : sync_id_tensor_index;
            int32_t tmp = (int32_t)ttnn_tensor[ptr][sync_id_tensor_index];
            tile_val = wrap_index(tmp, input_shape[dim_index]);
            ptr--;
        } else {
            tile_val = is_last_two_dim ? (begins[dim_index] + 31) / 32 + tile_idx
                                       : begins[dim_index] + tile_idx * steps[dim_index];
        }
        res += tile_val * accum_prod_origin_tiles;

        if (!is_ttnn_tensor[dim_index] || !flag) {
            accum_prod_set_tiles = is_last_two_dim ? accum_prod_set_tiles * ((input_set_shape[dim_index] + 31) / 32)
                                                   : accum_prod_set_tiles * input_set_shape[dim_index];
            flag = 1;
        }
        accum_prod_origin_tiles = is_last_two_dim ? accum_prod_origin_tiles * ((input_shape[dim_index] + 31) / 32)
                                                  : accum_prod_origin_tiles * input_shape[dim_index];
    }
    return res;
}

/**
 * @brief Maps a stick ID from the slice to the corresponding stick ID in the value tensor.
 *
 * This function is used to find the correct source stick in the `value` tensor when its shape
 * is broadcastable to the shape of the slice being set. It calculates the destination stick ID
 * in the `value` tensor for a given source `stick_id` from the slice.
 *
 * The calculation iterates from the inner-most to outer-most dimensions (excluding the last dimension,
 * which is stick-level). It converts the linear `stick_id` into a multi-dimensional index based on
 * `input_set_shape`. This index is then mapped to the `value` tensor's space using a modulo operation
 * with `value_shape` for each dimension, effectively handling broadcasting. The resulting multi-dimensional
 * index in the `value` space is then converted back to a linear stick ID.
 *
 * @param stick_id The ID of the stick within the slice being set.
 * @param is_ttnn_tensor A boolean array indicating if a dimension is indexed by a tensor.
 * @param input_set_shape Shape of the slice being set.
 * @param value_shape Shape of the value tensor being broadcasted.
 * @param tensor_rank The rank of the tensors.
 * @return The calculated stick ID in the value tensor.
 */
ALWI uint32_t find_value_stick_id(
    uint32_t stick_id,
    tt_l1_ptr uint32_t* is_ttnn_tensor,
    tt_l1_ptr uint32_t* input_set_shape,
    tt_l1_ptr uint32_t* value_shape,
    uint32_t tensor_rank) {
    uint32_t i = 1;
    uint32_t accum_prod_input_set_shape = 1;
    uint32_t accum_prod_value_shape = 1;
    uint32_t res = 0;
    bool flag = false;
    while (i < tensor_rank) {
        i++;
        uint32_t dim_index = tensor_rank - i;
        res += ((stick_id / accum_prod_input_set_shape) % value_shape[dim_index]) * accum_prod_value_shape;
        if (!is_ttnn_tensor[dim_index] || !flag) {
            accum_prod_input_set_shape *= input_set_shape[dim_index];
            flag = 1;
        }
        accum_prod_value_shape *= value_shape[dim_index];
    }
    return res;
}

/**
 * @brief Maps a tile ID from the slice to the corresponding tile ID in the value tensor.
 *
 * This function is analogous to `find_value_stick_id` but operates on tiles. It is used to find the
 * correct source tile in the `value` tensor when its shape is broadcastable to the shape of the slice
 * being set. It calculates the destination tile ID in the `value` tensor for a given source `tile_id`
 * from the slice.
 *
 * The calculation iterates from the inner-most to outer-most dimensions. It converts the linear `tile_id`
 * into a multi-dimensional index based on the tile-based shape of the slice. This index is then mapped
 * to the `value` tensor's tile space using a modulo operation with the tile-based `value_shape` for each
 * dimension, effectively handling broadcasting. The resulting multi-dimensional index in the `value` space
 * is then converted back to a linear tile ID.
 *
 * For the last two dimensions, shapes are converted to tile-based shapes by dividing by the tile dimension (32).
 *
 * @param tile_id The ID of the tile within the slice being set.
 * @param is_ttnn_tensor A boolean array indicating if a dimension is indexed by a tensor.
 * @param input_set_shape Shape of the slice being set.
 * @param value_shape Shape of the value tensor being broadcasted.
 * @param tensor_rank The rank of the tensors.
 * @return The calculated tile ID in the value tensor.
 */
ALWI uint32_t find_value_tile_id(
    uint32_t tile_id,
    tt_l1_ptr uint32_t* is_ttnn_tensor,
    tt_l1_ptr uint32_t* input_set_shape,
    tt_l1_ptr uint32_t* value_shape,
    uint32_t tensor_rank) {
    uint32_t i = 0;
    uint32_t accum_prod_input_set_tiles = 1;
    uint32_t accum_prod_value_tiles = 1;
    uint32_t res = 0;
    bool flag = false;
    bool is_last_two_dim = true;
    while (i < tensor_rank) {
        i++;
        uint32_t dim_index = tensor_rank - i;
        if (i > 2) {
            is_last_two_dim = false;
        }

        if (is_last_two_dim) {
            res += ((tile_id / accum_prod_input_set_tiles) % ((value_shape[dim_index] + 31) / 32)) *
                   accum_prod_value_tiles;
        } else {
            res += ((tile_id / accum_prod_input_set_tiles) % value_shape[dim_index]) * accum_prod_value_tiles;
        }

        if (!is_ttnn_tensor[dim_index] || !flag) {
            accum_prod_input_set_tiles = is_last_two_dim
                                             ? accum_prod_input_set_tiles * ((input_set_shape[dim_index] + 31) / 32)
                                             : accum_prod_input_set_tiles * input_set_shape[dim_index];
            flag = 1;
        }
        accum_prod_value_tiles = is_last_two_dim ? accum_prod_value_tiles * ((value_shape[dim_index] + 31) / 32)
                                                 : accum_prod_value_tiles * value_shape[dim_index];
    }
    return res;
}

/**
 * Converts a bfloat16 value to float
 * @param bfp16_bits The bfloat16 value as uint16_t
 * @return The equivalent float value
 */
ALWI float bfloat16_to_float(uint16_t bfp16_bits) {
    uint32_t float_bits = static_cast<uint32_t>(bfp16_bits) << 16;
    float result;
    std::memcpy(&result, &float_bits, sizeof(result));
    return result;
}
