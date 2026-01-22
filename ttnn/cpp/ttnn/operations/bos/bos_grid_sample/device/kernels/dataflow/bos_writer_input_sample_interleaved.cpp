#include "dataflow_api.h"
#include "debug/dprint.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

void kernel_main() {
    // matrix A (src0) has shape of BC, HiWi => should be HEIGHT_SHARDED or INTERLEAVED with row stick (BHiWi, 1)
    // matrix B (src1) has shape of BHoWo, 2 => should be HEIGHT_SHARDED or INTERLEAVED with col stick (1, 2K^2).
    // dst has shape of: BHoWo, C
    // In this file, we do INTERLEAVED reading
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t dst_addr = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t num_sticks = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(1);
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr uint32_t out_stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t out_block_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_input_width_blocks = get_compile_time_arg_val(5);

    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_nbytes};

    for (uint32_t stick_num = start_id; stick_num < start_id + num_sticks; stick_num++) {
        uint64_t dst_noc_addr = get_noc_addr(stick_num, s_out);
        // Can I read continuously write out_block_size as bilinear upsample into dst without risk of override other
        // tensor? Idk, but just in case...
        uint32_t total_size = out_stick_nbytes;
        uint32_t write_bytes = std::min(out_block_size, total_size);
        for (uint32_t i = 0; i < num_input_width_blocks; i++) {
            // DPRINT << "write_bytes: " << write_bytes << ENDL();
            cb_wait_front(cb_id_out, 1);
            uint32_t cb_out_addr = get_write_ptr(cb_id_out);
            noc_async_write(cb_out_addr, dst_noc_addr, write_bytes);
            dst_noc_addr += write_bytes;
            total_size = total_size - write_bytes;
            write_bytes = std::min(out_block_size, total_size);
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, 1);
        }
    }
}
