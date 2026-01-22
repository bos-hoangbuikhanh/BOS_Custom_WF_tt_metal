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

// Zero out all tiles for a given circular buffer.
template <uint32_t cb_id>
FORCE_INLINE void zero_out_tiles() {
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    static_assert(
        tile_size % MEM_ZEROS_SIZE == 0, "Tile size must be a multiple of MEM_ZEROS_BASE for zeroing out tiles");
    const uint32_t num_tiles = get_local_cb_interface(cb_id).fifo_num_pages;
    const uint32_t num_zeros_reads = (tile_size / MEM_ZEROS_SIZE) * num_tiles;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t write_addr = get_write_ptr(cb_id);

    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state<true>(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    noc_async_read_barrier();
}


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
    uint32_t start_stick_id   = get_arg_val<uint32_t>(5);
    uint32_t num_units_per_core = get_arg_val<uint32_t>(6);

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

    //others
    constexpr bool value_is_dram              = (bool)get_compile_time_arg_val(9);
    constexpr bool shapes_is_dram             = (bool)get_compile_time_arg_val(10);
    constexpr bool grid_is_dram               = (bool)get_compile_time_arg_val(11);
    constexpr bool weight_tensor_is_dram      = (bool)get_compile_time_arg_val(12);
    constexpr bool bilinear_weight_hash_is_dram = (bool)get_compile_time_arg_val(13);

    constexpr uint32_t value_cb_pagesize      = get_compile_time_arg_val(14);
    constexpr uint32_t value_stick_nbytes     = get_compile_time_arg_val(15);
    constexpr uint32_t bilinear_weight_hash_unit_size = get_compile_time_arg_val(16);

    constexpr uint32_t batch_size             = get_compile_time_arg_val(17);
    constexpr uint32_t num_queries            = get_compile_time_arg_val(18);
    constexpr uint32_t num_heads              = get_compile_time_arg_val(19);
    constexpr uint32_t num_keys               = get_compile_time_arg_val(20);
    constexpr uint32_t num_levels             = get_compile_time_arg_val(21);
    constexpr uint32_t num_points             = get_compile_time_arg_val(22);
    constexpr uint32_t bilinear_steps_x       = get_compile_time_arg_val(23);
    constexpr uint32_t bilinear_steps_y       = get_compile_time_arg_val(24);
    constexpr uint32_t bilinear_hash_size     = get_compile_time_arg_val(25);

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

    zero_out_tiles<value_zero_cbi>();
    uint32_t value_zero_addr = get_read_ptr(value_zero_cbi);

    uint32_t weight_tensor_read_addr = get_read_ptr(weight_tensor_cbi);

    volatile tt_l1_ptr uint16_t level_start_stick[num_levels];
    level_start_stick[0] = 0;

    //precompute flatten tile idx
    volatile tt_l1_ptr int flatten_weight_idx[32];
    volatile tt_l1_ptr int flatten_grid_idx[32];

    for (uint32_t i = 0; i < 32; i++){
        flatten_weight_idx[i] = ((i >> 4) << 8) + (i & 15u);
        int row = ((i >> 4) << 4) + ((i >> 1) & 7u); // i/2/8 * 16 + (i/2)%8
        int col = (i & 1u) << 4; //i%2 * 16
        flatten_grid_idx[i] = (row << 5) + col;    //row*32 + col
    }

    bool flag1 = false, flag2 = false;
    uint32_t end = start_stick_id + num_units_per_core;
    uint32_t num_accum_points = num_levels*num_points;
    uint32_t lvl_points_n_tiles = (num_accum_points + 31) / 32;
    int16_t in_height, in_width;
    for (; start_stick_id < end; start_stick_id++){
        uint32_t bid = start_stick_id / (num_heads * num_queries);
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

                // -- start denorm grid --
                cb_reserve_back(grid_cbi, 1);
                cb_reserve_back(flipped_shapes_cbi, 1);

                uint32_t grid_write_addr = get_write_ptr(grid_cbi);
                flipped_shapes_write_addr = get_write_ptr(flipped_shapes_cbi);
                noc_async_read_tile(tile_id, grid, grid_write_addr);
                if (!flag1 || lvl_points_n_tiles>1){
                    flag1=true;
                    noc_async_read_tile(tile_offset, flipped_shapes, flipped_shapes_write_addr);
                }
                noc_async_read_barrier();

                cb_push_back(grid_cbi, 1);
                cb_push_back(flipped_shapes_cbi, 1);

                // -- now we have denormalized coords --
                cb_wait_front(floored_grid_cbi, 1);
                floored_grid_read_addr = get_read_ptr(floored_grid_cbi);
                cb_pop_front(floored_grid_cbi, 1);
            }

#ifdef use_fp32
            volatile tt_l1_ptr float* floored_grid_stick = reinterpret_cast<volatile tt_l1_ptr float*>(floored_grid_read_addr);
            volatile tt_l1_ptr float* delta_stick = reinterpret_cast<volatile tt_l1_ptr float*>(delta_read_addr);
            volatile tt_l1_ptr float* weight_tensor_stick = reinterpret_cast<volatile tt_l1_ptr float*>(weight_tensor_read_addr);
#else
            volatile tt_l1_ptr uint16_t* floored_grid_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(floored_grid_read_addr);
            volatile tt_l1_ptr uint16_t* delta_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(delta_read_addr);
            volatile tt_l1_ptr uint16_t* weight_tensor_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(weight_tensor_read_addr);
#endif
            volatile tt_l1_ptr uint16_t* flipped_shapes_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(flipped_shapes_write_addr);

            // weight stick is divided such that each row has maximum 16 elements. The spill is in 8th row
            int get_flatten_grid_idx = flatten_grid_idx[idx];

            if (i%num_points==0){
                // level increased
                in_width = bfloat16_to_int16(flipped_shapes_stick[get_flatten_grid_idx]);
                in_height = bfloat16_to_int16(flipped_shapes_stick[get_flatten_grid_idx + 1]);
                if (!flag2 && level_id>0){
                    level_start_stick[level_id] = level_start_stick[level_id-1] + in_height*in_width;
                }
            }
            uint32_t id = bid*num_keys*num_heads + level_start_stick[level_id] * num_heads + head_id;

#ifdef use_fp32
            int16_t x_low = (int16_t)(floored_grid_stick[get_flatten_grid_idx]);
            int16_t y_low = (int16_t)(floored_grid_stick[get_flatten_grid_idx + 1]);
#else
            int16_t x_low = bfloat16_to_int16(floored_grid_stick[get_flatten_grid_idx]);
            int16_t y_low = bfloat16_to_int16(floored_grid_stick[get_flatten_grid_idx + 1]);
#endif
            int16_t x_high = x_low + 1;
            int16_t y_high = y_low + 1;
            int16_t y_low_offset = y_low * in_width;
            int16_t y_high_offset = y_high * in_width;

            cb_reserve_back(value_cbi, 4);

            uint32_t value_addr = get_write_ptr(value_cbi);
            tt::data_movement::common::tt_memmove<false, false, false, 0>(value_addr, value_zero_addr, value_cb_pagesize*4);

            uint32_t tmp_value_addr = value_addr;

            uint64_t noc_addr;
            if (y_low >= 0 && x_low >= 0 && y_low <= (in_height - 1) && x_low <= (in_width - 1)){
                uint32_t key_id = y_low_offset + x_low;
                noc_addr = value.get_noc_addr(id + key_id*num_heads);
                noc_async_read(noc_addr, tmp_value_addr, value_cb_pagesize);
            }
            tmp_value_addr += value_cb_pagesize;
            
            if (x_low >= 0 && y_high <= (in_height - 1) && y_high >= 0 && x_low <= (in_width - 1) ){
                uint32_t key_id = y_high_offset + x_low;
                noc_addr = value.get_noc_addr(id + key_id*num_heads);
                noc_async_read(noc_addr, tmp_value_addr, value_cb_pagesize);
            }
            tmp_value_addr += value_cb_pagesize;

            if (y_low >= 0 && x_high <= (in_width - 1) && y_low <= (in_height - 1) && x_high >= 0){
                uint32_t key_id = y_low_offset + x_high;
                noc_addr = value.get_noc_addr(id + key_id*num_heads);
                noc_async_read(noc_addr, tmp_value_addr, value_cb_pagesize);
            }
            tmp_value_addr += value_cb_pagesize;

            if (x_high <= (in_width - 1) && y_high <= (in_height - 1) && x_high >= 0 && y_high >= 0){
                uint32_t key_id = y_high_offset + x_high;
                noc_addr = value.get_noc_addr(id + key_id*num_heads);
                noc_async_read(noc_addr, tmp_value_addr, value_cb_pagesize);
            }

            // if (i == (num_accum_points-1)) noc_async_read_barrier();// #TODO: check this
            noc_async_read_barrier();
            cb_push_back(value_cbi, 4);
        }

        // all the level_start_id are cached
        flag2 = true;
    }
}
