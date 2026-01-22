// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"
#include "debug/dprint.h"

template <uint32_t in_ntiles_c>
inline void reduce_h_fused(const uint32_t in_cb_id, const uint32_t in_scalar_cb_id, const uint32_t out_cb_id) {
    cb_reserve_back(out_cb_id, 1);
    tile_regs_acquire();
    cb_wait_front(in_cb_id, 4);
    unpack_tilizeA_B_block(
        in_cb_id,
        in_scalar_cb_id,
        in_ntiles_c,
        0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/,
        2 /* unpack 2 faces ) */
    );
    for (uint32_t c_i = 0; c_i < in_ntiles_c; ++c_i) {
        // reduce 2 faces, write to index c_i of destination (out cb)
        reduce_tile_math(c_i, 2);
    }
    cb_pop_front(in_cb_id, 4);

    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dest<in_ntiles_c>(out_cb_id, 1, 0, 1, 2); /* pack 1 row (1x32) */
    tile_regs_release();
    cb_push_back(out_cb_id, 1);
}

namespace NAMESPACE {
void MAIN {
    uint32_t nsticks_per_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t in_scalar_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(2);

    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(3);
    constexpr uint32_t out_ntiles_c = get_compile_time_arg_val(4);
    constexpr uint32_t num_input_width_blocks =
        get_compile_time_arg_val(5);  // 1 block contain (MAX_TILES_PER_REDUCTION * 32) elements

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;
    constexpr uint32_t num_output_tiles = out_ntiles_c;  //* nblocks;

    // @TODO: double circular buffer i.e. when processing first cb, read/write other cb.
    // neginf source A is false and keep dimension after reduction
    // icb0, scaler, MAX_TILES_PER_REDUCTION=8, out_cb_id, 2 faces, each face has 4 rows
    tilizeA_B_reduce_init<false, true>(in_cb_id, in_scalar_cb_id, max_tiles_per_iter, out_cb_id, 2, 4);
    // 2 faces, each face has 1 row
    pack_untilize_dest_init<num_output_tiles>(out_cb_id, 1, 2);
    for (uint32_t i = 0; i < nsticks_per_core; i++) {
        for (uint32_t j = 0; j < num_input_width_blocks - 1; j++) {
            cb_wait_front(in_scalar_cb_id, 1);
            reduce_h_fused<max_tiles_per_iter>(in_cb_id, in_scalar_cb_id, out_cb_id);
            cb_pop_front(in_scalar_cb_id, 1);
        }
        cb_wait_front(in_scalar_cb_id, 1);
        reduce_h_fused<partial_iter_output_tiles>(in_cb_id, in_scalar_cb_id, out_cb_id);
        cb_pop_front(in_scalar_cb_id, 1);
    }
}  // MAIN
}  // namespace NAMESPACE
