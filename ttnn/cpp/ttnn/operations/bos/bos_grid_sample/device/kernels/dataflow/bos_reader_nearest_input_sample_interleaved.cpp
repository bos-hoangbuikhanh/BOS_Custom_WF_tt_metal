#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#define ALWI inline __attribute__((always_inline))

ALWI float bfloat16_to_float(uint16_t bfp16_bits) {
    uint32_t float_bits = static_cast<uint32_t>(bfp16_bits) << 16;
    float result;
    std::memcpy(&result, &float_bits, sizeof(result));
    return result;
}

ALWI int32_t iround_even_manual(float v) {
    float intpart;
    float frac = std::fabs(std::modf(v, &intpart));

    if (frac < 0.5f) {
        return static_cast<int32_t>(intpart);
    } else if (frac > 0.5f) {
        return static_cast<int32_t>(intpart + (v >= 0 ? 1.0f : -1.0f));
    } else {
        int32_t i = static_cast<int32_t>(intpart);
        if (i % 2 == 0) {
            return i;
        } else {
            return i + (v >= 0 ? 1 : -1);
        }
    }
}

void kernel_main() {
    DeviceZoneScopedN("Reader Input Sample Nearest Interleaved");
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

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src0_addr, .page_size = src0_stick_nbytes};
    const InterleavedAddrGen<src1_is_dram> s1 = {.bank_base_address = src1_addr, .page_size = src1_stick_nbytes};
    uint32_t ptr1_write_addr = get_write_ptr(cb_id_in1);
    for (uint32_t stick_num = 0; stick_num < num_sticks; stick_num++) {
        int32_t b = start_id / (out_height * out_width);

        cb_reserve_back(cb0_id_in0, 1);
        uint32_t ptr1_write_addr = get_write_ptr(cb_id_in1);
        noc_async_read_page(start_id, s1, ptr1_write_addr);

        volatile tt_l1_ptr float* src1_stick = reinterpret_cast<volatile tt_l1_ptr float*>(ptr1_write_addr);
        noc_async_read_barrier();

        float x = src1_stick[0];
        float y = src1_stick[1];

        if (align_corners == 1) {
            x = ((x + 1) * (in_width - 1.f)) / 2.f;
            y = ((y + 1) * (in_height - 1.f)) / 2.f;
        } else {
            x = ((x + 1) * in_width - 1.f) / 2.f;  // == ( (x+1)/2 * in_width - 0.5 )
            y = ((y + 1) * in_height - 1.f) / 2.f;
        }
        uint32_t in_scalar_write_addr = get_write_ptr(cb_id_scalar);
        uint32_t ptr00_write_addr = get_write_ptr(cb0_id_in0);
        uint32_t ptr01_write_addr, ptr02_write_addr, ptr03_write_addr;

        volatile tt_l1_ptr float* scalar = reinterpret_cast<volatile tt_l1_ptr float*>(in_scalar_write_addr);

        /* 1. integer pixel coordinates */
        int32_t ix = iround_even_manual(x);
        int32_t iy = iround_even_manual(y);

        /* 2. is the point inside the valid input rectangle? */
        const bool inside = (ix >= 0 && ix < int32_t(in_width)) && (iy >= 0 && iy < int32_t(in_height));

        float w1;

        if (inside) {
            /* 3a. in-bounds → clamp to edge (safety) and DMA one pixel */
            const int32_t max_x = static_cast<int32_t>(in_width - 1);
            const int32_t max_y = static_cast<int32_t>(in_height - 1);

            ix = std::max<int32_t>(0, std::min<int32_t>(ix, max_x));
            iy = std::max<int32_t>(0, std::min<int32_t>(iy, max_y));

            w1 = 1.0f;  // single contributing pixel
            noc_async_read_page(b * in_height * in_width + iy * in_width + ix, s0, ptr00_write_addr);
        } else {
            /* 3b. out-of-bounds → zero contribution, no DMA */
            w1 = 0.0f;
        }

        /* 4. hand the mask to the writer and keep the pipeline balanced */
        scalar[0] = w1;
        noc_async_read_barrier();

        cb_push_back(cb_id_scalar, 1);
        cb_push_back(cb0_id_in0, 1);  // push even when no DMA to stay in-step
        start_id++;
    }
}
