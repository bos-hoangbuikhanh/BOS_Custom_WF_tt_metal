// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint_pages.h"

void kernel_main() {
    // Compile-time constants for configuration
    constexpr uint32_t shard_cb_id = get_compile_time_arg_val(0);
    constexpr bool read_from_dram = get_compile_time_arg_val(1);
    constexpr bool unaligned = get_compile_time_arg_val(2);
    constexpr uint32_t unit_size = get_compile_time_arg_val(3);
    constexpr uint32_t local_unit_size_padded = get_compile_time_arg_val(4);
    constexpr uint32_t remote_unit_size_padded = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scratch_index = get_compile_time_arg_val(6);

    // Runtime arguments
    uint32_t src_addr = get_arg_val<uint32_t>(0);                       // Source address
    uint32_t write_offset = get_arg_val<uint32_t>(1);                   // Offset for writing
    uint32_t num_reads = get_arg_val<uint32_t>(2);                      // Number of read operations
    tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(3));  // Pointer to additional arguments
    uint32_t args_idx = 0;                                              // Index for iterating through args

    // Compute the initial write address in L1 memory
    uint32_t l1_write_addr = get_write_ptr(shard_cb_id) + write_offset;

    if constexpr (unaligned) {
        // Handle unaligned data transfers
        uint32_t l1_scratch_write_addr = get_write_ptr(cb_scratch_index);  // Scratchpad write address
        uint32_t l1_scratch_read_addr = get_read_ptr(cb_scratch_index);    // Scratchpad read address

        for (uint32_t i = 0; i < num_reads; ++i) {
            // Read arguments for each transfer
            uint32_t bank_id = args[args_idx++];                               // Bank ID
            uint32_t src_offset = args[args_idx++];                            // Source offset
            uint32_t addr = src_addr + src_offset;                             // Compute source address
            uint32_t units_to_transfer = args[args_idx++];                     // Number of units to transfer
            uint32_t read_size = units_to_transfer * remote_unit_size_padded;  // Total read size

            // Perform asynchronous read from NoC to scratchpad
            noc_async_read(get_noc_addr_from_bank_id<read_from_dram>(bank_id, addr), l1_scratch_write_addr, read_size);
            noc_async_read_barrier();  // Wait for read to complete

            // Align and transfer data from scratchpad to L1 memory
            uint64_t pad_align_noc_addr = get_noc_addr(l1_scratch_read_addr);
            for (uint32_t j = 0; j < units_to_transfer; ++j) {
                noc_async_read(pad_align_noc_addr, l1_write_addr, unit_size);  // Transfer unit
                l1_write_addr += unit_size;                                    // Advance write address
                pad_align_noc_addr += remote_unit_size_padded;                 // Advance read address
            }
            noc_async_read_barrier();  // Wait for all transfers to complete
        }
    } else {
        // Handle aligned data transfers
        for (uint32_t i = 0; i < num_reads; ++i) {
            // Read arguments for each transfer
            uint32_t bank_id = args[args_idx++];                 // Bank ID
            uint32_t addr = src_addr + args[args_idx++];         // Compute source address
            uint32_t units_to_transfer = args[args_idx++];       // Number of units to transfer
            uint32_t read_size = units_to_transfer * unit_size;  // Total read size

            // Perform direct asynchronous read from NoC to L1 memory
            noc_async_read(get_noc_addr_from_bank_id<read_from_dram>(bank_id, addr), l1_write_addr, read_size);
            l1_write_addr += read_size;  // Advance write address
        }
        noc_async_read_barrier();  // Wait for all transfers to complete
    }
}
