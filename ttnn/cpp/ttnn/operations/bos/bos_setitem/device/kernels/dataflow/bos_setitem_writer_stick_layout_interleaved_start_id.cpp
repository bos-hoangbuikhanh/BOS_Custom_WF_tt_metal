// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/bos/bos_setitem/device/kernels/dataflow/bos_setitem_common.hpp"

#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t input_cbi = get_compile_time_arg_val(0);
    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t input_shape_last_dim = get_compile_time_arg_val(2);
    constexpr uint32_t element_size = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_rank = get_compile_time_arg_val(4);
    constexpr uint32_t n_ttnn_tensor = get_compile_time_arg_val(5);

    const uint32_t input_address = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks_per_core = get_arg_val<uint32_t>(2);
    tt_l1_ptr uint32_t* input_shape = (tt_l1_ptr uint32_t*)(get_arg_addr(3));
    tt_l1_ptr uint32_t* input_set_shape = input_shape + tensor_rank;
    tt_l1_ptr uint32_t* begins = input_set_shape + tensor_rank;
    tt_l1_ptr uint32_t* steps = begins + tensor_rank;
    tt_l1_ptr uint32_t* is_ttnn_tensor = steps + tensor_rank;
    tt_l1_ptr uint32_t* indexes_address = is_ttnn_tensor + tensor_rank;
    tt_l1_ptr uint32_t* indexes_page_size = indexes_address + n_ttnn_tensor;
    tt_l1_ptr uint32_t* indexes_is_dram = indexes_page_size + n_ttnn_tensor;
    volatile tt_l1_ptr uint32_t* ttnn_tensor[n_ttnn_tensor];

    const uint32_t input_page_size = input_shape_last_dim * element_size;
    const InterleavedAddrGen<input_is_dram> input = {.bank_base_address = input_address, .page_size = input_page_size};

    uint32_t nxt_cbi = input_cbi + 3;
    for (uint32_t i = 0; i < n_ttnn_tensor; i++) {
        bool tensor_is_dram = indexes_is_dram[i] == 1;
        uint32_t page_size = indexes_page_size[i];
        uint64_t index_noc_addr;
        if (tensor_is_dram) {
            const InterleavedAddrGen<true> index = {.bank_base_address = indexes_address[i], .page_size = page_size};
            index_noc_addr = get_noc_addr(0, index);
        } else {
            const InterleavedAddrGen<false> index = {.bank_base_address = indexes_address[i], .page_size = page_size};
            index_noc_addr = get_noc_addr(0, index);
        }
        uint32_t tensor_index_l1_addr = get_read_ptr(nxt_cbi);
        noc_async_read(index_noc_addr, tensor_index_l1_addr, page_size);
        ttnn_tensor[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tensor_index_l1_addr);
        nxt_cbi++;
    }
    noc_async_read_barrier();

    for (uint32_t stick_id = start_id; stick_id < start_id + num_sticks_per_core; stick_id++) {
        uint32_t input_bank_id = find_input_stick_id(
            stick_id,
            input_shape,
            input_set_shape,
            begins,
            steps,
            is_ttnn_tensor,
            ttnn_tensor,
            n_ttnn_tensor,
            tensor_rank);
        const uint32_t input_cbi_ = input_cbi + (stick_id % 2);
        cb_wait_front(input_cbi_, 1);
        uint32_t input_l1_addr = get_read_ptr(input_cbi_);
        uint64_t input_noc_addr = get_noc_addr(input_bank_id, input);
        noc_async_write(input_l1_addr, input_noc_addr, input_page_size);
        noc_async_write_barrier();
        cb_pop_front(input_cbi_, 1);
    }
}
