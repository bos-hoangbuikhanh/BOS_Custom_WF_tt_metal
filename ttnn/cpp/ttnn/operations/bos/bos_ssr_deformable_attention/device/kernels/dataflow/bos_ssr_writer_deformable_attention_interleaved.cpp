#include <cmath>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    uint32_t dst_addr   = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id   = get_arg_val<uint32_t>(1);
    uint32_t num_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out                 = get_compile_time_arg_val(0);
    constexpr bool     dst_is_dram               = (bool)get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_pagesize           = get_compile_time_arg_val(2);
    constexpr uint32_t num_input_width_blocks    = get_compile_time_arg_val(3); //currently not used
    constexpr uint32_t out_stick_nbytes          = get_compile_time_arg_val(4);
    constexpr uint32_t batch_size                = get_compile_time_arg_val(5);
    constexpr uint32_t num_queries               = get_compile_time_arg_val(6);
    constexpr uint32_t num_heads                 = get_compile_time_arg_val(7);
    constexpr uint32_t num_total_sticks          = get_compile_time_arg_val(8);


    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_nbytes};

    uint32_t stick_id = start_tile_id*32;
    uint32_t end = start_tile_id+num_units;
    for (; start_tile_id < end; start_tile_id++){
        for (uint32_t row = 0; row < 32 && stick_id < num_total_sticks; row++, stick_id++){
#ifdef is_QHB
            uint32_t query_id = stick_id / num_heads;
            uint32_t head_id = stick_id % num_heads;
            uint32_t bid = 0;
#else
            uint32_t query_id = (stick_id / num_heads) % num_queries;
            uint32_t bid = stick_id / (num_heads * num_queries);
            uint32_t head_id = stick_id % num_heads;
#endif
            cb_wait_front(cb_id_out, 1);
            uint32_t cb_out_addr = get_read_ptr(cb_id_out);

#ifdef is_padded_input
            uint32_t id_bqh = ((bid * num_queries) + query_id) * num_heads + head_id;
            uint32_t output_stick_id = id_bqh;
            uint64_t dst_noc_addr = get_noc_addr(output_stick_id, s_out);
            noc_async_write(cb_out_addr, dst_noc_addr, out_cb_pagesize);
#else
            uint32_t output_stick_id = bid*num_queries + query_id;
            uint64_t dst_noc_addr = get_noc_addr(output_stick_id, s_out);
            noc_async_write(cb_out_addr, dst_noc_addr + head_id*out_cb_pagesize, out_cb_pagesize);
#endif
            noc_async_write_barrier();
            cb_pop_front(cb_id_out, 1);
        }
    }
}
