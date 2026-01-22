// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/bos/bos_setitem/device/kernels/dataflow/bos_setitem_common.hpp"

#include "debug/dprint.h"

#define ALWI inline __attribute__((always_inline))

constexpr uint32_t inner_slices_size = get_compile_time_arg_val(4);
template <typename T, uint32_t unroll_factor>
FORCE_INLINE void broadcast_last_dim(
    volatile tt_l1_ptr T* inp, volatile tt_l1_ptr uint32_t* ttnn_tensor, uint32_t input_shape_last_dim, T val) {
#pragma GCC unroll inner_slices_size
    for (uint32_t i = 0; i < unroll_factor; i++) {
        uint32_t inp_idx = wrap_index((int32_t)ttnn_tensor[i], input_shape_last_dim);
        inp[inp_idx] = val;
    }
}

template <typename T, uint32_t unroll_factor>
FORCE_INLINE void broadcast_last_dim(volatile tt_l1_ptr T* inp, uint32_t begin, uint32_t step, T val) {
#pragma GCC unroll inner_slices_size
    for (uint32_t i = 0; i < unroll_factor; i++) {
        inp[begin + step * i] = val;
    }
}

void kernel_main() {
    constexpr uint32_t input_cbi = get_compile_time_arg_val(0);
    constexpr uint32_t value_cbi = get_compile_time_arg_val(1);
    constexpr bool input_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool value_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t input_shape_last_dim = get_compile_time_arg_val(5);
    constexpr uint32_t value_shape_last_dim = get_compile_time_arg_val(6);
    constexpr uint32_t tensor_rank = get_compile_time_arg_val(7);
    constexpr uint32_t element_size = get_compile_time_arg_val(8);
    constexpr bool is_broadcast_last_dim = get_compile_time_arg_val(9) == 1;
    constexpr uint32_t n_ttnn_tensor = get_compile_time_arg_val(10);

    const uint32_t input_address = get_arg_val<uint32_t>(0);
    const uint32_t value_address = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);  // the id of the slice index to start from
    const uint32_t num_sticks_per_core = get_arg_val<uint32_t>(3);
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

    const uint32_t input_page_size = input_shape_last_dim * element_size;
    const uint32_t value_page_size = value_shape_last_dim * element_size;
    const InterleavedAddrGen<input_is_dram> input = {.bank_base_address = input_address, .page_size = input_page_size};
    const InterleavedAddrGen<value_is_dram> value = {.bank_base_address = value_address, .page_size = value_page_size};

    uint32_t nxt_cbi = value_cbi + 1;
    for (uint32_t i = 0; i < n_ttnn_tensor; i++) {
        uint32_t page_size = indexes_page_size[i];
        uint64_t index_noc_addr;
        if (indexes_is_dram[i]) {
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

    uint32_t value_l1_addr = get_write_ptr(value_cbi);
    uint32_t offset = begins[tensor_rank - 1] * element_size;

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
        uint32_t value_bank_id =
            find_value_stick_id(stick_id, is_ttnn_tensor, input_set_shape, value_shape, tensor_rank);
        const uint32_t input_cbi_ = input_cbi + (stick_id % 2);
        cb_reserve_back(input_cbi_, 1);
        uint32_t input_l1_addr = get_write_ptr(input_cbi_);
        uint64_t input_noc_addr = get_noc_addr(input_bank_id, input);
        uint64_t value_noc_addr = get_noc_addr(value_bank_id, value);
        noc_async_read(input_noc_addr, input_l1_addr, input_page_size);
        noc_async_read(value_noc_addr, value_l1_addr, value_page_size);
        noc_async_read_barrier();
        if (is_broadcast_last_dim) {
            switch (element_size) {
                case 2: {  // float16, bfloat16
                    uint16_t value = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(value_l1_addr)[0];
                    volatile tt_l1_ptr uint16_t* inp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_l1_addr);
                    if (is_ttnn_tensor[tensor_rank - 1]) {
                        broadcast_last_dim<uint16_t, inner_slices_size>(
                            inp, ttnn_tensor[n_ttnn_tensor - 1], input_shape_last_dim, value);
                    } else {
                        broadcast_last_dim<uint16_t, inner_slices_size>(
                            inp, begins[tensor_rank - 1], steps[tensor_rank - 1], value);
                    }
                    break;
                }
                case 4: {  // float32, int
                    uint32_t value = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(value_l1_addr)[0];
                    volatile tt_l1_ptr uint32_t* inp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_l1_addr);
                    if (is_ttnn_tensor[tensor_rank - 1]) {
                        broadcast_last_dim<uint32_t, inner_slices_size>(
                            inp, ttnn_tensor[n_ttnn_tensor - 1], input_shape_last_dim, value);
                    } else {
                        broadcast_last_dim<uint32_t, inner_slices_size>(
                            inp, begins[tensor_rank - 1], steps[tensor_rank - 1], value);
                    }
                    break;
                }
            }
        } else {
            // copy data from value cb to input cb
            tt::data_movement::common::tt_memmove<false, false, false, 0>(
                input_l1_addr + offset, value_l1_addr, value_page_size);
        }
        cb_push_back(input_cbi_, 1);
    }
}
