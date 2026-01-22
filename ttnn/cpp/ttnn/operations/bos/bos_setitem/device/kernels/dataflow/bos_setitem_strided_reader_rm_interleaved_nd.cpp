// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/bos/bos_setitem/device/kernels/dataflow/bos_setitem_common.hpp"

#define ALWI inline __attribute__((always_inline))


constexpr uint32_t inner_slices_size = get_compile_time_arg_val(4);
template <typename T, uint32_t unroll_factor>
FORCE_INLINE void broadcast_last_dim(
    volatile tt_l1_ptr T* inp, 
    volatile tt_l1_ptr uint32_t* ttnn_tensor,
    uint32_t input_shape_last_dim,
    T val){
#pragma GCC unroll inner_slices_size
    for (uint32_t i = 0; i < unroll_factor; i++){
        uint32_t inp_idx = wrap_index((int32_t)ttnn_tensor[i], input_shape_last_dim);
        inp[inp_idx] = val;
    }
}

template <typename T, uint32_t unroll_factor>
FORCE_INLINE void broadcast_last_dim(
    volatile tt_l1_ptr T* inp, 
    uint32_t begin, 
    uint32_t step, 
    T val){
#pragma GCC unroll inner_slices_size
    for (uint32_t i = 0; i < unroll_factor; i++) inp[begin+step*i] = val;
}


constexpr uint32_t value_shape_last_dim = get_compile_time_arg_val(6);
template <typename T, uint32_t unroll_factor>
ALWI void selective_assign_value_to_input(
    volatile tt_l1_ptr T* inp,
    volatile tt_l1_ptr uint32_t* ttnn_tensor,
    uint32_t input_shape_last_dim,
    volatile tt_l1_ptr T* value
){
#pragma GCC unroll value_shape_last_dim
    for (uint32_t i = 0; i < unroll_factor; i++){
        uint32_t inp_idx = wrap_index((int32_t)ttnn_tensor[i], input_shape_last_dim);
        inp[inp_idx] = value[i];
    }
}

template <typename T, uint32_t unroll_factor>
ALWI void selective_assign_value_to_input(
    volatile tt_l1_ptr T* inp,
    uint32_t begin,
    uint32_t step,
    volatile tt_l1_ptr T* value
){
#pragma GCC unroll value_shape_last_dim
    for (uint32_t i = 0; i < unroll_factor; i++) inp[begin + i*step] = value[i];
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
    const uint32_t start_id = get_arg_val<uint32_t>(2); // the id of the slice index to start from
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

    const uint32_t input_page_size = input_shape_last_dim*element_size;
    const uint32_t value_page_size = value_shape_last_dim*element_size;
    const InterleavedAddrGen<input_is_dram> input = {.bank_base_address = input_address, .page_size = input_page_size};
    const InterleavedAddrGen<value_is_dram> value = {.bank_base_address = value_address, .page_size = value_page_size};

    uint32_t nxt_cbi = value_cbi+1;
    for (uint32_t i = 0; i < n_ttnn_tensor; i++){
        bool tensor_is_dram = indexes_is_dram[i] == 1;
        uint32_t page_size = indexes_page_size[i];
        uint64_t index_noc_addr;
        if (tensor_is_dram){
            const InterleavedAddrGen<true> index = {.bank_base_address = indexes_address[i], .page_size = page_size};
            index_noc_addr = get_noc_addr(0, index);
        }
        else{
            const InterleavedAddrGen<false> index = {.bank_base_address = indexes_address[i], .page_size = page_size};
            index_noc_addr = get_noc_addr(0, index);
        }
        uint32_t tensor_index_l1_addr = get_write_ptr(nxt_cbi);
        noc_async_read(index_noc_addr, tensor_index_l1_addr, page_size);
        ttnn_tensor[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tensor_index_l1_addr);
        nxt_cbi++;
    }
    noc_async_read_barrier();

    // Check how many element in indices is valid
    uint32_t valid_len_last = value_shape_last_dim;  // default cap
    if (n_ttnn_tensor > 0 && is_ttnn_tensor[tensor_rank - 1]) {
        const uint32_t last_i = n_ttnn_tensor - 1;
        volatile tt_l1_ptr int32_t* signed_indices =
            reinterpret_cast<volatile tt_l1_ptr int32_t*>(ttnn_tensor[last_i]);

        const uint32_t total_idx_elems = indexes_page_size[last_i] / sizeof(int32_t);

        uint32_t w = 0;
        for (uint32_t r = 0; r < total_idx_elems; ++r) { // Loop through indices
            // DPRINT << r << ENDL();
            int32_t v = signed_indices[r]; 
            if (v == -1) continue;    
            signed_indices[w++] = v;
        }
        valid_len_last = w;

        // DPRINT << "Compacted last-dim indices: valid_len=" << valid_len_last
        //     << " (from " << total_idx_elems << ")" << ENDL();
    }

    uint32_t value_l1_addr = get_write_ptr(value_cbi);
    for (uint32_t stick_id = start_id; stick_id < start_id + num_sticks_per_core; ++stick_id) {
    
        uint32_t input_bank_id = find_input_stick_id(
            stick_id,
            input_shape,
            input_set_shape,
            begins,
            steps,
            is_ttnn_tensor,
            ttnn_tensor,
            n_ttnn_tensor,
            tensor_rank
        );
        uint32_t value_bank_id = find_value_stick_id(stick_id, is_ttnn_tensor, input_set_shape, value_shape, tensor_rank);

        // DPRINT << "Printing input_set_shape: " << input_set_shape[0] << ENDL();
        // DPRINT << "Printing value_shape: " << value_shape[0] << ENDL();

        const uint32_t input_cbi_ = input_cbi + (stick_id%2);
        cb_reserve_back(input_cbi_, 1);
        uint32_t input_l1_addr = get_write_ptr(input_cbi_);
        uint64_t input_noc_addr = get_noc_addr(input_bank_id, input);
        uint64_t value_noc_addr = get_noc_addr(value_bank_id, value);
        noc_async_read(input_noc_addr, input_l1_addr, input_page_size);
        noc_async_read(value_noc_addr, value_l1_addr, value_page_size);
        noc_async_read_barrier();
        if (is_broadcast_last_dim) {
            switch (element_size) {
                case 2: {
                    const uint16_t scalar = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(value_l1_addr)[0];
                    volatile tt_l1_ptr uint16_t* inp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_l1_addr);
                    if (is_ttnn_tensor[tensor_rank - 1]) {
                        volatile tt_l1_ptr int32_t* idx =
                            reinterpret_cast<volatile tt_l1_ptr int32_t*>(ttnn_tensor[n_ttnn_tensor - 1]);
                        const uint32_t eff = valid_len_last;  // 
                        for (uint32_t i = 0; i < eff; ++i) {
                            const uint32_t widx = wrap_index(idx[i], input_shape_last_dim);
                            inp[widx] = scalar;
                        }
                    } else {
                        const uint32_t begin = begins[tensor_rank - 1];
                        const uint32_t step  = steps[tensor_rank - 1];
                        for (uint32_t i = 0; i < value_shape_last_dim; ++i) inp[begin + i * step] = scalar;
                    }
                    break;
                }
                case 4: {
                    const uint32_t scalar = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(value_l1_addr)[0];
                    volatile tt_l1_ptr uint32_t* inp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_l1_addr);
                    if (is_ttnn_tensor[tensor_rank - 1]) {
                        volatile tt_l1_ptr int32_t* idx =
                            reinterpret_cast<volatile tt_l1_ptr int32_t*>(ttnn_tensor[n_ttnn_tensor - 1]);
                        const uint32_t eff = valid_len_last;
                        for (uint32_t i = 0; i < eff; ++i) {
                            const uint32_t widx = wrap_index(idx[i], input_shape_last_dim);
                            inp[widx] = scalar;
                        }
                    } else {
                        const uint32_t begin = begins[tensor_rank - 1];
                        const uint32_t step  = steps[tensor_rank - 1];
                        for (uint32_t i = 0; i < value_shape_last_dim; ++i) inp[begin + i * step] = scalar;
                    }
                    break;
                }
            }
        }

        else { // non-broadcast
            switch (element_size) {
                case 2: {
                    volatile tt_l1_ptr uint16_t* val = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(value_l1_addr);
                    volatile tt_l1_ptr uint16_t* inp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_l1_addr);

                    if (is_ttnn_tensor[tensor_rank - 1]) {
                        volatile tt_l1_ptr int32_t* idx =
                            reinterpret_cast<volatile tt_l1_ptr int32_t*>(ttnn_tensor[n_ttnn_tensor - 1]);

                        uint32_t eff = valid_len_last;
                        if (eff > value_shape_last_dim) eff = value_shape_last_dim;

                        for (uint32_t i = 0; i < eff; ++i) {
                            const uint32_t widx = wrap_index(idx[i], input_shape_last_dim);
                            inp[widx] = val[i];
                        }
                    } else {
                        const uint32_t begin = begins[tensor_rank - 1];
                        const uint32_t step  = steps[tensor_rank - 1];
                        for (uint32_t i = 0; i < value_shape_last_dim; ++i) inp[begin + i * step] = val[i];
                    }
                    break;
                }
                case 4: {
                    volatile tt_l1_ptr uint32_t* val = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(value_l1_addr);
                    volatile tt_l1_ptr uint32_t* inp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_l1_addr);

                    if (is_ttnn_tensor[tensor_rank - 1]) {
                        volatile tt_l1_ptr int32_t* idx =
                            reinterpret_cast<volatile tt_l1_ptr int32_t*>(ttnn_tensor[n_ttnn_tensor - 1]);

                        uint32_t eff = valid_len_last;
                        if (eff > value_shape_last_dim) eff = value_shape_last_dim;

                        for (uint32_t i = 0; i < eff; ++i) {
                            const uint32_t widx = wrap_index(idx[i], input_shape_last_dim);
                            inp[widx] = val[i];
                        }
                    } else {
                        const uint32_t begin = begins[tensor_rank - 1];
                        const uint32_t step  = steps[tensor_rank - 1];
                        for (uint32_t i = 0; i < value_shape_last_dim; ++i) inp[begin + i * step] = val[i];
                    }
                    break;
                }
            }
        }


        cb_push_back(input_cbi_, 1);

    }
}