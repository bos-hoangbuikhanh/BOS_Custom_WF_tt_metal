// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    // in0 tensor args
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t in0_tensor_stride_w = get_arg_val<uint32_t>(2);
    uint32_t in0_tensor_stride_h = get_arg_val<uint32_t>(3);
    uint32_t in0_tensor_next_block_stride = get_arg_val<uint32_t>(4);

    // in0 block args
    uint32_t in0_block_w = get_arg_val<uint32_t>(5);
    uint32_t in0_block_h = get_arg_val<uint32_t>(6);
    uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(7);

    // in0 mcast args
    uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(17);
    uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(18);
    uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(19);
    uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(20);
    uint32_t in0_mcast_num_dests = get_arg_val<uint32_t>(21);
    uint32_t in0_mcast_sender_noc_x = get_arg_val<uint32_t>(22);
    uint32_t in0_mcast_sender_noc_y = get_arg_val<uint32_t>(23);
    uint32_t in0_mcast_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(24));
    uint32_t in0_mcast_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(25));

    // batch args
    uint32_t MtKt = get_arg_val<uint32_t>(35);  // if 0
    uint32_t KtNt = get_arg_val<uint32_t>(36);
    uint32_t batch = get_arg_val<uint32_t>(37);
    uint32_t bcast_B = get_arg_val<uint32_t>(38);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    volatile tt_l1_ptr uint32_t* in0_mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_mcast_receiver_semaphore_addr);

    // Operand 0
    cb_reserve_back(cb_id_in0, in0_block_num_tiles);

    // Set in0 semaphore value to INVALID
    noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, INVALID);

    // Atomic increment source core counter
    uint64_t in0_mcast_sender_semaphore_noc_addr =
        get_noc_addr(in0_mcast_sender_noc_x, in0_mcast_sender_noc_y, in0_mcast_sender_semaphore_addr);
    (in0_mcast_sender_semaphore_noc_addr, 1);

    // wait on in0 semaphore value to become VALID (set by mcast sender after it multicasts data)
    noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, VALID);

    cb_push_back(cb_id_in0, in0_block_num_tiles);
}
