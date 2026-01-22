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

#define ALWI inline __attribute__((always_inline))

ALWI float bfloat16_to_float(uint16_t bfp16_bits) {
    uint32_t float_bits = static_cast<uint32_t>(bfp16_bits) << 16;
    float result;
    std::memcpy(&result, &float_bits, sizeof(result));
    return result;
}

ALWI int16_t bfloat16_to_int16(uint16_t bfp16_bits) {
    float f = bfloat16_to_float(bfp16_bits);
    return static_cast<int16_t>(f);
}

ALWI uint16_t float_to_bfloat16(float value) {
    uint32_t float_bits;
    std::memcpy(&float_bits, &value, sizeof(float));
    return static_cast<uint16_t>(float_bits >> 16);
}

void kernel_main() {
    // Runtime arguments
    uint32_t value_addr  = get_arg_val<uint32_t>(0);
    uint32_t grid_addr = get_arg_val<uint32_t>(1);
    uint32_t flipped_shapes_addr  = get_arg_val<uint32_t>(2);
    uint32_t weight_tensor_addr  = get_arg_val<uint32_t>(3);
    uint32_t bilinear_weight_hash_addr  = get_arg_val<uint32_t>(4);
    uint32_t dst_addr   = get_arg_val<uint32_t>(5);
    uint32_t start_stick_id   = get_arg_val<uint32_t>(6);
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
    //support CBs
    constexpr uint32_t weight_tensor_cbi    = get_compile_time_arg_val(8);
    // output CBs
    constexpr uint32_t cb_id_out              = get_compile_time_arg_val(9);

    //others
    constexpr bool value_is_dram              = (bool)get_compile_time_arg_val(10);
    constexpr bool shapes_is_dram             = (bool)get_compile_time_arg_val(11);
    constexpr bool grid_is_dram               = (bool)get_compile_time_arg_val(12);
    constexpr bool weight_tensor_is_dram      = (bool)get_compile_time_arg_val(13);
    constexpr bool bilinear_weight_hash_is_dram = (bool)get_compile_time_arg_val(14);
    constexpr bool dst_is_dram                = get_compile_time_arg_val(15);

    constexpr uint32_t value_cb_pagesize      = get_compile_time_arg_val(16);
    constexpr uint32_t value_stick_nbytes     = get_compile_time_arg_val(17);
    constexpr uint32_t bilinear_weight_hash_unit_size = get_compile_time_arg_val(18);
    constexpr uint32_t num_input_width_blocks = get_compile_time_arg_val(19);
    constexpr uint32_t out_stick_nbytes  = get_compile_time_arg_val(20);

    constexpr uint32_t batch_size             = get_compile_time_arg_val(21);
    constexpr uint32_t num_queries            = get_compile_time_arg_val(22);
    constexpr uint32_t num_heads              = get_compile_time_arg_val(23);
    constexpr uint32_t num_keys               = get_compile_time_arg_val(24);
    constexpr uint32_t num_levels             = get_compile_time_arg_val(25);
    constexpr uint32_t num_points             = get_compile_time_arg_val(26);
    constexpr uint32_t bilinear_steps_x       = get_compile_time_arg_val(27);
    constexpr uint32_t bilinear_steps_y       = get_compile_time_arg_val(28);
    constexpr uint32_t bilinear_hash_size     = get_compile_time_arg_val(29);

    constexpr uint32_t grid_tile_size = get_tile_size(grid_cbi);
    constexpr uint32_t weight_tile_size = get_tile_size(weight_tensor_cbi);
    constexpr uint32_t shapes_tile_size = get_tile_size(flipped_shapes_cbi);
    constexpr DataFormat grid_data_format = get_dataformat(grid_cbi);
    constexpr DataFormat weight_data_format = get_dataformat(weight_tensor_cbi);
    constexpr DataFormat shapes_data_format = get_dataformat(flipped_shapes_cbi);

    const InterleavedAddrGen<value_is_dram> value = {.bank_base_address = value_addr, .page_size = value_stick_nbytes};
    const InterleavedAddrGenFast<grid_is_dram> grid = {.bank_base_address = grid_addr, .page_size = grid_tile_size, .data_format = grid_data_format};
    const InterleavedAddrGenFast<shapes_is_dram> flipped_shapes = {.bank_base_address = flipped_shapes_addr, .page_size = shapes_tile_size, .data_format = shapes_data_format};
    const InterleavedAddrGenFast<weight_tensor_is_dram> weight_tensor = {.bank_base_address = weight_tensor_addr, .page_size = weight_tile_size, .data_format = weight_data_format};
    const InterleavedAddrGen<bilinear_weight_hash_is_dram> bilinear_weight_hash = {.bank_base_address = bilinear_weight_hash_addr, .page_size = bilinear_weight_hash_unit_size};
    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_nbytes};
    
    uint32_t weight_tensor_read_addr = get_read_ptr(weight_tensor_cbi);

    //precompute flatten tile idx
    volatile tt_l1_ptr int flatten_weight_idx[32];
    volatile tt_l1_ptr int flatten_grid_idx[32];

    for (uint32_t i = 0; i < 32; i++){
        flatten_weight_idx[i] = ((i >> 4) << 8) + (i & 15u);
        int row = ((i >> 4) << 4) + ((i >> 1) & 7u); // i/2/8 * 16 + (i/2)%8
        int col = (i & 1u) << 4; //i%2 * 16
        flatten_grid_idx[i] = (row << 5) + col;    //row*32 + col
    }

    uint32_t end = start_stick_id + num_units_per_core;
    uint32_t num_accum_points = num_levels*num_points;
    uint32_t lvl_points_n_tiles = (num_accum_points + 31) / 32;
    int16_t in_height, in_width;
    for (; start_stick_id < end; start_stick_id++){
        uint32_t bid = start_stick_id / (num_heads * num_queries);
        uint32_t query_id = (start_stick_id / num_heads) % num_queries;
        uint32_t head_id = start_stick_id % num_heads;
        // Get ptrs for circular buffers
        uint32_t floored_grid_read_addr, delta_read_addr, flipped_shapes_write_addr;

        for (uint32_t i = 0; i < num_accum_points; i++){
            int level_id = i/num_points;
            int point_id = i%num_points;
            uint32_t idx = i & 31u; // i%32

            if (idx == 0){
                // need to process new tile
                uint32_t tile_offset = i/32;
                uint32_t tile_id = start_stick_id*lvl_points_n_tiles + tile_offset;

                // -- read attention weight --
                noc_async_read_tile(tile_id, weight_tensor, weight_tensor_read_addr);
                noc_async_read_barrier();

                cb_wait_front(delta_cbi, 1);
                delta_read_addr = get_read_ptr(delta_cbi);
                cb_pop_front(delta_cbi, 1);
            }

#ifdef use_fp32
            volatile tt_l1_ptr float* delta_stick = reinterpret_cast<volatile tt_l1_ptr float*>(delta_read_addr);
            volatile tt_l1_ptr float* weight_tensor_stick = reinterpret_cast<volatile tt_l1_ptr float*>(weight_tensor_read_addr);
#else
            volatile tt_l1_ptr uint16_t* delta_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(delta_read_addr);
            volatile tt_l1_ptr uint16_t* weight_tensor_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(weight_tensor_read_addr);
#endif
            // weight stick is divided such that each row has maximum 16 elements. The spill is in 8th row
            int get_flatten_grid_idx = flatten_grid_idx[idx];

#ifdef use_fp32
            int16_t qx   = (int16_t)(delta_stick[get_flatten_grid_idx]);
            int16_t qy   = (int16_t)(delta_stick[get_flatten_grid_idx+1]);
#else
            int16_t qx   = bfloat16_to_int16(delta_stick[get_flatten_grid_idx]);
            int16_t qy   = bfloat16_to_int16(delta_stick[get_flatten_grid_idx+1]);
#endif
            if (qx == bilinear_steps_y) qx = 0;
            uint32_t hash_idx = qx * bilinear_steps_y + qy;

            cb_reserve_back(bilinear_weight_cbi, 1);
            cb_reserve_back(weight_cbi, 1);
            uint32_t weight_write_addr = get_write_ptr(weight_cbi);
            uint32_t bilinear_weight_write_addr = get_write_ptr(bilinear_weight_cbi);

#ifdef use_fp32
            volatile tt_l1_ptr float* weight = reinterpret_cast<volatile tt_l1_ptr float*>(weight_write_addr);
#else
            volatile tt_l1_ptr uint16_t* weight = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(weight_write_addr);
#endif

            // Read bilinear weight from hash tensor
            uint64_t noc_addr = bilinear_weight_hash.get_noc_addr(hash_idx);
            noc_async_read(noc_addr, bilinear_weight_write_addr, bilinear_weight_hash_unit_size);
            weight[0] = weight_tensor_stick[flatten_weight_idx[idx]];
            noc_async_read_barrier();

            cb_push_back(bilinear_weight_cbi, 1);
            cb_push_back(weight_cbi, 1);
        }
        // Write
        cb_wait_front(cb_id_out, 1);
        uint32_t cb_out_addr = get_read_ptr(cb_id_out);
#ifdef is_padded_input
        uint32_t output_stick_id = start_stick_id;
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
}
