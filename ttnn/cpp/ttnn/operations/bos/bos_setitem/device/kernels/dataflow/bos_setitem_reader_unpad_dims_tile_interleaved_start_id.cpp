// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/bos/bos_setitem/device/kernels/dataflow/bos_setitem_common.hpp"

void kernel_main() {
    constexpr uint32_t value_cbi = get_compile_time_arg_val(0);
    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool value_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t tensor_rank = get_compile_time_arg_val(3);
    constexpr uint32_t n_ttnn_tensor = get_compile_time_arg_val(4);

    const uint32_t input_address = get_arg_val<uint32_t>(0);
    const uint32_t value_address = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);  // the id of the slice index to start from
    const uint32_t num_tiles_per_core = get_arg_val<uint32_t>(3);
    tt_l1_ptr uint32_t* input_shape = (tt_l1_ptr uint32_t*)(get_arg_addr(4));
    tt_l1_ptr uint32_t* input_set_shape = input_shape + tensor_rank;
    tt_l1_ptr uint32_t* value_shape = input_set_shape + tensor_rank;
    tt_l1_ptr uint32_t* begins = value_shape + tensor_rank;
    tt_l1_ptr uint32_t* steps = begins + tensor_rank;
    tt_l1_ptr uint32_t* is_ttnn_tensor = steps + tensor_rank;
    tt_l1_ptr uint32_t* indexes_address = is_ttnn_tensor + tensor_rank;
    tt_l1_ptr uint32_t* indexes_page_size = indexes_address + n_ttnn_tensor;
    tt_l1_ptr uint32_t* indexes_is_dram = indexes_page_size + n_ttnn_tensor;
    volatile tt_l1_ptr uint32_t* ttnn_tensor[n_ttnn_tensor];

    constexpr uint32_t tile_size = get_tile_size(value_cbi);
    constexpr DataFormat data_format = get_dataformat(value_cbi);

    const InterleavedAddrGenFast<value_is_dram> value = {
        .bank_base_address = value_address, .page_size = tile_size, .data_format = data_format};
    const InterleavedAddrGenFast<input_is_dram> input = {
        .bank_base_address = input_address, .page_size = tile_size, .data_format = data_format};

    for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles_per_core; tile_id++) {
        uint32_t value_tile_id = find_value_tile_id(tile_id, is_ttnn_tensor, input_set_shape, value_shape, tensor_rank);
        uint32_t tmp = tile_id % 2;
        cb_reserve_back(value_cbi + tmp, 1);
        uint32_t src_buffer_l1_addr = get_write_ptr(value_cbi + tmp);
        noc_async_read_tile(value_tile_id, value, src_buffer_l1_addr);
        noc_async_read_barrier();
        cb_push_back(value_cbi + tmp, 1);
    }
}
