#include "dataflow_api.h"
#include "debug/dprint.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#define ALWI inline __attribute__((always_inline))

ALWI float uint32_to_float(uint32_t f) {
    float ret;
    std::memcpy(&ret, &f, sizeof(float));
    return ret;
}

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

void nearest(
    volatile tt_l1_ptr float* inp0,
    volatile tt_l1_ptr float* scalar,
    volatile tt_l1_ptr float* output,
    const uint32_t in_channels) {
    const float w = scalar[0];  // 1 → keep, 0 → zero
    if (w == 0.0f) {
        for (uint32_t i = 0; i < in_channels; ++i) {
            // output[i] = float_to_bfloat16(0.0f);
            output[i] = 0.0f;
        }
    } else {  // w == 1.0f
        for (uint32_t i = 0; i < in_channels; ++i) {
            output[i] = inp0[i];
        }
    }
};

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

    constexpr uint32_t cb0_id_in0 = get_compile_time_arg_val(0);          // [0]
    constexpr uint32_t cb1_id_in0 = get_compile_time_arg_val(1);          // [1]
    constexpr uint32_t cb2_id_in0 = get_compile_time_arg_val(2);          // [2]
    constexpr uint32_t cb3_id_in0 = get_compile_time_arg_val(3);          // [3]
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(4);           // [4]
    constexpr uint32_t cb_id_scalar = get_compile_time_arg_val(5);        // [5]
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(6);           // [6]
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(7);      // [7]
    constexpr bool src1_is_dram = (bool)get_compile_time_arg_val(8);      // [8]
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(9);       // [9]
    constexpr uint32_t src0_stick_nbytes = get_compile_time_arg_val(10);  // all sticks at once [10]
    constexpr uint32_t src1_stick_nbytes = get_compile_time_arg_val(11);  // per core [11]
    constexpr uint32_t out_stick_nbytes = get_compile_time_arg_val(12);   // [12]
    constexpr uint32_t in_channels = get_compile_time_arg_val(13);        // [13]
    constexpr uint32_t in_height = get_compile_time_arg_val(14);          // [14]
    constexpr uint32_t in_width = get_compile_time_arg_val(15);           // [15]
    constexpr uint32_t out_height = get_compile_time_arg_val(16);         // [16]
    constexpr uint32_t out_width = get_compile_time_arg_val(17);          // [17]
    constexpr bool align_corners = (bool)get_compile_time_arg_val(18);    // [18]

    const InterleavedAddrGen<dst_is_dram> s_out = {.bank_base_address = dst_addr, .page_size = out_stick_nbytes};
    uint32_t cb_out_addr = get_write_ptr(cb_id_out);
    volatile tt_l1_ptr float* out_stick = reinterpret_cast<volatile tt_l1_ptr float*>(cb_out_addr);

    for (uint32_t stick_num = 0; stick_num < num_sticks; stick_num++) {
        cb_wait_front(cb0_id_in0, 1);
        cb_wait_front(cb_id_scalar, 1);

        uint32_t scalar_read_addr = get_read_ptr(cb_id_scalar);
        volatile tt_l1_ptr float* scalar = reinterpret_cast<volatile tt_l1_ptr float*>(scalar_read_addr);
        uint32_t ptr00_read_addr = get_read_ptr(cb0_id_in0);
        volatile tt_l1_ptr float* src00_stick = reinterpret_cast<volatile tt_l1_ptr float*>(ptr00_read_addr);
        nearest(src00_stick, scalar, out_stick, in_channels);

        uint64_t dst_noc_addr = get_noc_addr(start_id, s_out);
        noc_async_write(cb_out_addr, dst_noc_addr, out_stick_nbytes);
        noc_async_write_barrier();

        cb_pop_front(cb0_id_in0, 1);
        cb_pop_front(cb_id_scalar, 1);
        start_id++;
    }
}
