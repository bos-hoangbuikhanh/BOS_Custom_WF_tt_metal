
#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_unary/rounding.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

namespace NAMESPACE {
void MAIN {
    uint32_t ntiles_per_core = get_arg_val<uint32_t>(0);

    // prepcompute CBs
    constexpr uint32_t grid_cbi = get_compile_time_arg_val(0);
    constexpr uint32_t flipped_shapes_cbi = get_compile_time_arg_val(1);
    //interm CBs
    constexpr uint32_t interm_floored_grid_cbi = get_compile_time_arg_val(2);
    // precomputed result CBs
    constexpr uint32_t denormed_cbi = get_compile_time_arg_val(3);
    constexpr uint32_t floored_grid_cbi = get_compile_time_arg_val(4);
    constexpr uint32_t delta_cbi = get_compile_time_arg_val(5);
    // bilinear
    constexpr uint32_t value_cbi = get_compile_time_arg_val(6);
    constexpr uint32_t weight_cbi = get_compile_time_arg_val(7);
    constexpr uint32_t bilinear_weight_cbi = get_compile_time_arg_val(8);
    constexpr uint32_t scaler_cbi = get_compile_time_arg_val(9);
    // bilineared result CB
    constexpr uint32_t out_cbi = get_compile_time_arg_val(10);
    //metadata
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(11);
    constexpr uint32_t out_ntiles_c = get_compile_time_arg_val(12);
    constexpr uint32_t num_input_width_blocks = get_compile_time_arg_val(13);   // 1 block contain (MAX_TILES_PER_REDUCTION * 32) elements
    constexpr uint32_t num_levels = get_compile_time_arg_val(14);
    constexpr uint32_t num_points = get_compile_time_arg_val(15);
    constexpr uint32_t step_x = get_compile_time_arg_val(16);
    constexpr uint32_t step_y = get_compile_time_arg_val(17);
    constexpr uint32_t num_rows = get_compile_time_arg_val(18);
    constexpr uint32_t batch_size = get_compile_time_arg_val(19);
	constexpr uint32_t accum_batch_cbi = get_compile_time_arg_val(20);

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;
    constexpr uint32_t num_output_tiles = out_ntiles_c;  //* nblocks;

    uint32_t num_accum_points = batch_size*num_levels*num_points;
    for (uint32_t i = 0; i < ntiles_per_core; i++) {
        // denormed = ttnn::multiply_(denormalized_sampling_locations, flipped) - 0.5f
        binary_op_init_common(grid_cbi, flipped_shapes_cbi, denormed_cbi);
        reconfig_data_format(grid_cbi, flipped_shapes_cbi);   //because they may have different DF
        pack_reconfig_data_format(denormed_cbi);

        cb_wait_front(grid_cbi, 1);
        cb_wait_front(flipped_shapes_cbi, 1);
        cb_reserve_back(floored_grid_cbi, 1);
        cb_reserve_back(interm_floored_grid_cbi, 1);
        cb_reserve_back(denormed_cbi, 1);
        ACQ();
#ifdef is_denormed_grid
        copy_tile_to_dst_init_short(grid_cbi);
        copy_tile(grid_cbi, 0, 0);  //copy grid to dst[0]
        pack_tile(0, denormed_cbi);
        cb_push_back(denormed_cbi, 1);
#else
        mul_tiles_init(grid_cbi, flipped_shapes_cbi);
        mul_tiles(grid_cbi, flipped_shapes_cbi, 0, 0, 0); //dst[0] = denormed
        fill_tile(1, 0.5f);     //dst[1] = 0.5f
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);  //denormed -= 0.5f
        pack_tile(0, denormed_cbi);
        cb_push_back(denormed_cbi, 1);
#endif  // is_denormed_grid

        // floored = ttnn::floor(denormed)
        copy_dest_values(2, 0); //copy values from dst[0] to dst[2]
        rounding_op_tile_init();
#ifdef use_fp32
        floor_tile_float32(2);
#else
        floor_tile(2);
#endif
        pack_tile(2, floored_grid_cbi);
        pack_tile(2, interm_floored_grid_cbi);
        cb_push_back(floored_grid_cbi, 1);
        cb_push_back(interm_floored_grid_cbi, 1);
        REL();
        cb_pop_front(flipped_shapes_cbi, 1);
        cb_pop_front(grid_cbi, 1);

        // delta = ttnn::subtract(denorm, floored)
        binary_op_init_common(denormed_cbi, interm_floored_grid_cbi, delta_cbi);
        reconfig_data_format(denormed_cbi, interm_floored_grid_cbi);   //because they may have different DF
        pack_reconfig_data_format(delta_cbi);
        sub_tiles_init(denormed_cbi, interm_floored_grid_cbi);
        cb_wait_front(interm_floored_grid_cbi, 1);
        cb_wait_front(denormed_cbi, 1);
        cb_reserve_back(delta_cbi, 1);
        ACQ();
        sub_tiles(denormed_cbi, interm_floored_grid_cbi, 0, 0, 0);
#ifdef use_bilinear_weight_hash
        fill_tile(1, (float)step_x);     //dst[1] = step_x
        mul_binary_tile(0, 1, 0);             //delta_x = delta * step_x
#ifdef use_fp32
        floor_tile_float32(0);
#else
        floor_tile(0);
#endif
#endif
        pack_tile(0, delta_cbi);
        cb_push_back(delta_cbi, 1);
        REL();
        cb_pop_front(interm_floored_grid_cbi, 1);
        cb_pop_front(denormed_cbi, 1);
        // Prepare denormalized sampling locations
        for (uint32_t row = 0; row < num_rows; row++){
            for (uint32_t col = 0; col < num_accum_points; col++){
                //------------------------------------------------
                // ---BILINEAR---
                binary_op_init_common(bilinear_weight_cbi, weight_cbi, scaler_cbi);
                // Basically all subsequent SFPU operations require tile layout for correct execution.
                // In the MSDA (or general bilinear) case, only 4 elements are needed for bilinear weight and 1 for attention weight,
                // which matches the tile's data layout. Therefore, explicit tilization is not necessary here.
                reconfig_data_format(bilinear_weight_cbi, weight_cbi);
                pack_reconfig_data_format(scaler_cbi);

                mul_tiles_bcast_scalar_init_short(bilinear_weight_cbi, weight_cbi);
                cb_reserve_back(scaler_cbi, 1);
                cb_wait_front(bilinear_weight_cbi, 1);
                cb_wait_front(weight_cbi, 1);

                tile_regs_acquire();
                mul_tiles_bcast_scalar(bilinear_weight_cbi, weight_cbi, 0, 0, 0);

                cb_pop_front(bilinear_weight_cbi, 1);
                cb_pop_front(weight_cbi, 1);

                tile_regs_commit();
                tile_regs_wait();

                pack_tile(0, scaler_cbi);

                tile_regs_release();
                cb_push_back(scaler_cbi, 1);

            }
			//bilinear and accumulate
            tilizeA_B_reduce_init<false, true>(value_cbi, scaler_cbi, in_ntiles_c, out_cbi, 2, 4);
			pack_untilize_dest_init<in_ntiles_c>(out_cbi, 1, 2);
            cb_reserve_back(out_cbi, 1);
            tile_regs_acquire();
            for (uint32_t col = 0; col < num_accum_points; col++){
                cb_wait_front(value_cbi, 4);
                cb_wait_front(scaler_cbi, 1);
                
                // Value * scalar
                unpack_tilizeA_B_block<false, true, false, true>(value_cbi, scaler_cbi, in_ntiles_c, 0, 2, 4);

                for (uint32_t c_i = 0; c_i < in_ntiles_c; ++c_i)
                    reduce_tile_math(c_i, 2);

                cb_pop_front(value_cbi, 4);
                cb_pop_front(scaler_cbi, 1);
            }

            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<in_ntiles_c>(out_cbi, 1, 0, 1, 2);
            tile_regs_release();
            cb_push_back(out_cbi, 1);
            pack_untilize_uninit(out_cbi);
        }
    }
}
}  // namespace NAMESPACE
