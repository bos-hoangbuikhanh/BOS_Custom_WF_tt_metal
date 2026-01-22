#include "dataflow_api.h"
#include "debug/dprint.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#include "tools/profiler/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/bos/bos_ssr_deformable_attention/device/kernels/utils.hpp"

void kernel_main() {
    // Runtime arguments
    uint32_t value_addr  = get_arg_val<uint32_t>(0);
    uint32_t grid_addr = get_arg_val<uint32_t>(1);
    uint32_t flipped_shapes_addr  = get_arg_val<uint32_t>(2);
    uint32_t weight_tensor_addr  = get_arg_val<uint32_t>(3);
    uint32_t bilinear_weight_hash_addr  = get_arg_val<uint32_t>(4);
    uint32_t dst_addr   = get_arg_val<uint32_t>(5);
    uint32_t start_tile_id   = get_arg_val<uint32_t>(6);
    uint32_t num_units_per_core = get_arg_val<uint32_t>(7);

    // prepcompute CBs
    constexpr uint32_t grid_cbi             = get_compile_time_arg_val(0);
    constexpr uint32_t flipped_shapes_cbi   = get_compile_time_arg_val(1);
    // precomputed result CBs
    constexpr uint32_t delta_cbi            = get_compile_time_arg_val(2);
    constexpr uint32_t floored_grid_cbi     = get_compile_time_arg_val(3);
    // bilinear CBs
    constexpr uint32_t value_cbi            = get_compile_time_arg_val(4);
    constexpr uint32_t value_zero_cbi       = get_compile_time_arg_val(5);
    constexpr uint32_t weight_cbi           = get_compile_time_arg_val(6);
    constexpr uint32_t bilinear_weight_cbi  = get_compile_time_arg_val(7);
    constexpr uint32_t tmp_bilinear_weight_cbi = get_compile_time_arg_val(8);
    //support CBs
    constexpr uint32_t weight_tensor_cbi    = get_compile_time_arg_val(9);
    // output CBs
    constexpr uint32_t cb_id_out              = get_compile_time_arg_val(10);

    //others
    constexpr bool value_is_dram              = (bool)get_compile_time_arg_val(11);
    constexpr bool shapes_is_dram             = (bool)get_compile_time_arg_val(12);
    constexpr bool grid_is_dram               = (bool)get_compile_time_arg_val(13);
    constexpr bool weight_tensor_is_dram      = (bool)get_compile_time_arg_val(14);
    constexpr bool bilinear_weight_hash_is_dram = (bool)get_compile_time_arg_val(15);
    constexpr bool dst_is_dram                = get_compile_time_arg_val(16);

    constexpr uint32_t value_cb_pagesize      = get_compile_time_arg_val(17);
    constexpr uint32_t value_stick_nbytes     = get_compile_time_arg_val(18);
    constexpr uint32_t bilinear_weight_hash_unit_size = get_compile_time_arg_val(19);
    constexpr uint32_t num_input_width_blocks = get_compile_time_arg_val(20);
    constexpr uint32_t out_stick_nbytes  = get_compile_time_arg_val(21);

    constexpr uint32_t batch_size             = get_compile_time_arg_val(22);
    constexpr uint32_t num_queries            = get_compile_time_arg_val(23);
    constexpr uint32_t num_heads              = get_compile_time_arg_val(24);
    constexpr uint32_t num_keys               = get_compile_time_arg_val(25);
    constexpr uint32_t num_levels             = get_compile_time_arg_val(26);
    constexpr uint32_t num_points             = get_compile_time_arg_val(27);
    constexpr uint32_t bilinear_steps_x       = get_compile_time_arg_val(28);
    constexpr uint32_t bilinear_steps_y       = get_compile_time_arg_val(29);
    constexpr uint32_t num_accum_points       = get_compile_time_arg_val(30);
    constexpr uint32_t num_total_sticks       = get_compile_time_arg_val(31);

    constexpr uint32_t weight_tile_size = get_tile_size(weight_tensor_cbi);
    constexpr DataFormat weight_data_format = get_dataformat(weight_tensor_cbi);

    uint32_t bilinear_w_esize = bilinear_weight_hash_unit_size/4;
    uint32_t bilinear_w_page_size = 4 * bilinear_steps_x * bilinear_steps_y * bilinear_w_esize;

    const InterleavedAddrGenFast<weight_tensor_is_dram> weight_tensor = {.bank_base_address = weight_tensor_addr, .page_size = weight_tile_size, .data_format = weight_data_format};
    const InterleavedAddrGen<bilinear_weight_hash_is_dram> bilinear_weight_hash = {.bank_base_address = bilinear_weight_hash_addr, .page_size = bilinear_w_page_size};
    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_nbytes};

    uint32_t weight_tensor_read_addr = get_read_ptr(weight_tensor_cbi);

    // delay 1 write
    bool delay = true;
    uint32_t MAX_POINTS_PER_REDUCTION = 4;
    uint32_t num_blocks = (num_accum_points + MAX_POINTS_PER_REDUCTION - 1) / MAX_POINTS_PER_REDUCTION;
    uint32_t stick_id = start_tile_id*32;

    cb_wait_front(value_zero_cbi, 1);
    cb_pop_front(value_zero_cbi, 1);

    uint32_t tmp_bilinear_w_write_addr = get_write_ptr(tmp_bilinear_weight_cbi);
    noc_async_read(bilinear_weight_hash.get_noc_addr(0), tmp_bilinear_w_write_addr, bilinear_w_page_size);

    int16_t in_height, in_width;
    uint32_t floored_grid_read_addr, delta_read_addr, flipped_shapes_write_addr, combined_scaler_write_addr;
    uint32_t end = start_tile_id + num_units_per_core;
    for (; start_tile_id < end; start_tile_id++){
        noc_async_read_tile(start_tile_id, weight_tensor, weight_tensor_read_addr);
        noc_async_read_barrier();

        cb_wait_front(delta_cbi, 1);
        delta_read_addr = get_read_ptr(delta_cbi);
        cb_pop_front(delta_cbi, 1);

#ifdef use_fp32
        volatile tt_l1_ptr float* delta_stick = reinterpret_cast<volatile tt_l1_ptr float*>(delta_read_addr);
        volatile tt_l1_ptr float* weight_tensor_stick = reinterpret_cast<volatile tt_l1_ptr float*>(weight_tensor_read_addr);
#else
        volatile tt_l1_ptr uint16_t* delta_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(delta_read_addr);
        volatile tt_l1_ptr uint16_t* weight_tensor_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(weight_tensor_read_addr);
#endif
        for (uint32_t row = 0; row < 32 && stick_id < num_total_sticks; row++, stick_id++){
            for (uint32_t block_idx = 0; block_idx < num_blocks; block_idx++){
                DeviceZoneScopedN("Read num_blocks weight");
                cb_reserve_back(bilinear_weight_cbi, 1);
                cb_reserve_back(weight_cbi, 1);
                uint32_t weight_write_addr = get_write_ptr(weight_cbi);
                uint32_t bilinear_weight_write_addr = get_write_ptr(bilinear_weight_cbi);
                uint32_t weight_id = 0;
                for (uint32_t point_idx = 0; point_idx < MAX_POINTS_PER_REDUCTION; point_idx++){
                    uint32_t global_point_idx = block_idx * MAX_POINTS_PER_REDUCTION + point_idx;
                    if (global_point_idx < num_accum_points){
                        uint32_t flatten_delta_idx = flatten_idx(row, global_point_idx, 2);
                        uint32_t flatten_weight_idx = flatten_idx(row, global_point_idx);
#ifdef use_fp32
                        int16_t qx = (int16_t)delta_stick[flatten_delta_idx];
                        int16_t qy = (int16_t)delta_stick[flatten_delta_idx + 1];
                        volatile tt_l1_ptr float* weight = reinterpret_cast<volatile tt_l1_ptr float*>(weight_write_addr);
#else
                        int16_t qx = bfloat16_to_int16(delta_stick[flatten_delta_idx]);
                        int16_t qy = bfloat16_to_int16(delta_stick[flatten_delta_idx + 1]);
                        volatile tt_l1_ptr uint16_t* weight = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(weight_write_addr);
#endif
                        if (qx == bilinear_steps_y) qx = 0;
                        uint32_t hash_idx = qx * bilinear_steps_y + qy;

                        auto weight_val = weight_tensor_stick[flatten_weight_idx];
                        weight[weight_id++] = weight_val;
                        weight[weight_id++] = weight_val;
                        weight[weight_id++] = weight_val;
                        weight[weight_id++] = weight_val;
                        // Read bilinear weight from hash tensor
                        tt::data_movement::common::tt_memmove<false, true, false, 0>(
                            bilinear_weight_write_addr, tmp_bilinear_w_write_addr + hash_idx*bilinear_weight_hash_unit_size, bilinear_weight_hash_unit_size);
                        bilinear_weight_write_addr += bilinear_weight_hash_unit_size;
                    }
                }
                cb_push_back(bilinear_weight_cbi, 1);
                cb_push_back(weight_cbi, 1);
            }

            if (delay){
                delay = false;
                continue;
            }

            // DeviceZoneScopedN("Write output");
            cb_wait_front(cb_id_out, 1);
            uint32_t cb_out_addr = get_read_ptr(cb_id_out);
            uint32_t write_stick_id = stick_id - 1;
#ifdef is_QHB
            uint32_t query_id = write_stick_id / num_heads;
            uint32_t head_id = write_stick_id % num_heads;
            uint32_t bid = 0;
            uint32_t layout_id = (bid * num_queries + query_id) * num_heads + head_id;

#else
            uint32_t query_id = (write_stick_id / num_heads) % num_queries;
            uint32_t bid = write_stick_id / (num_heads * num_queries);
            uint32_t head_id = write_stick_id % num_heads;
            uint32_t layout_id = write_stick_id;
#endif
#ifdef is_padded_input
            uint32_t output_stick_id = layout_id;
            uint64_t dst_noc_addr = get_noc_addr(output_stick_id, s_out);
            noc_async_write(cb_out_addr, dst_noc_addr, value_cb_pagesize);
#else
            uint32_t output_stick_id = bid*num_queries + query_id;
            uint64_t dst_noc_addr = get_noc_addr(output_stick_id, s_out);
            noc_async_write(cb_out_addr, dst_noc_addr + head_id*value_cb_pagesize, value_cb_pagesize);
#endif
            //TODO: cannot prove it is totally safe to disable this barrier
            // but in most cases it wont hurt since the read above is sync (which is very slow)
            // noc_async_write_barrier();
            cb_pop_front(cb_id_out, 1);
        }
    }

    cb_wait_front(cb_id_out, 1);
    uint32_t cb_out_addr = get_read_ptr(cb_id_out);
    uint32_t write_stick_id = stick_id - 1;
#ifdef is_QHB
    uint32_t query_id = write_stick_id / num_heads;
    uint32_t head_id = write_stick_id % num_heads;
    uint32_t bid = 0;
    uint32_t layout_id = (bid * num_queries + query_id) * num_heads + head_id;

#else
    uint32_t query_id = (write_stick_id / num_heads) % num_queries;
    uint32_t bid = write_stick_id / (num_heads * num_queries);
    uint32_t head_id = write_stick_id % num_heads;
    uint32_t layout_id = write_stick_id;
#endif
#ifdef is_padded_input
    uint32_t output_stick_id = layout_id;
    uint64_t dst_noc_addr = get_noc_addr(output_stick_id, s_out);
    noc_async_write(cb_out_addr, dst_noc_addr, value_cb_pagesize);
#else
    uint32_t output_stick_id = bid*num_queries + query_id;
    uint64_t dst_noc_addr = get_noc_addr(output_stick_id, s_out);
    noc_async_write(cb_out_addr, dst_noc_addr + head_id*value_cb_pagesize, value_cb_pagesize);
#endif
    noc_async_write_barrier();
    cb_pop_front(cb_id_out, 1);
}
