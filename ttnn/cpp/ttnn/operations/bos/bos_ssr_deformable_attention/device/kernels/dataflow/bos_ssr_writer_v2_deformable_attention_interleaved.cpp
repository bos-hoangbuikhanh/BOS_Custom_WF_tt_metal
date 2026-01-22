#include <cmath>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"
#define ALWI inline __attribute__((always_inline))

ALWI float bfloat16_to_float(uint16_t bfp16_bits) {
    uint32_t float_bits = static_cast<uint32_t>(bfp16_bits) << 16;
    float result;
    std::memcpy(&result, &float_bits, sizeof(result));
    return result;
}
ALWI uint16_t float_to_bfloat16(float value) {
    uint32_t float_bits;
    std::memcpy(&float_bits, &value, sizeof(float));
    return static_cast<uint16_t>(float_bits >> 16);
}
void kernel_main() {
    uint32_t dst_addr   = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id   = get_arg_val<uint32_t>(1);
    uint32_t num_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out       = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram   = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_pagesize  = get_compile_time_arg_val(2);
    constexpr uint32_t num_input_width_blocks = get_compile_time_arg_val(3); //currently not used
    constexpr uint32_t out_stick_nbytes  = get_compile_time_arg_val(4);
    constexpr uint32_t batch_size                = get_compile_time_arg_val(5);
    constexpr uint32_t num_queries               = get_compile_time_arg_val(6);
    constexpr uint32_t num_heads                 = get_compile_time_arg_val(7);
    constexpr uint32_t num_rows                  = get_compile_time_arg_val(8);

    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_nbytes};

    uint32_t end = start_tile_id+num_units;
    for (; start_tile_id < end; start_tile_id++){
        for (uint32_t row = 0; row < num_rows; row++){
            uint32_t linear_id = start_tile_id * 32 + row;
            uint32_t query_id = linear_id / num_heads;
            uint32_t head_id = linear_id % num_heads;
            cb_wait_front(cb_id_out, 1);
            uint32_t cb_out_addr = get_read_ptr(cb_id_out);
            volatile tt_l1_ptr uint16_t* out_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_out_addr);
#ifdef is_padded_input
            uint32_t output_stick_id = linear_id;
            uint64_t dst_noc_addr = get_noc_addr(output_stick_id, s_out);
            noc_async_write(cb_out_addr, dst_noc_addr, out_cb_pagesize);
#else       
            uint32_t output_stick_id = query_id;
            uint64_t dst_noc_addr = get_noc_addr(output_stick_id, s_out);
            noc_async_write(cb_out_addr, dst_noc_addr + head_id*out_cb_pagesize, out_cb_pagesize);
#endif

            noc_async_write_barrier();
            cb_pop_front(cb_id_out, 1);
        }
    }
}
