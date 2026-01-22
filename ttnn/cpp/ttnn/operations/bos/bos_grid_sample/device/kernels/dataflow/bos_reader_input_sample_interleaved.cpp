#include "dataflow_api.h"
#include "debug/dprint.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
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

ALWI uint16_t float_to_bfloat16(float value) {
    uint32_t float_bits;
    std::memcpy(&float_bits, &value, sizeof(float));
    return static_cast<uint16_t>(float_bits >> 16);
}

void kernel_main() {
    // matrix A (src0) has shape of BC, HiWi => should be HEIGHT_SHARDED or INTERLEAVED with row stick (BHiWi, 1)
    // matrix B (src1) has shape of BHoWo, 2 => should be HEIGHT_SHARDED or INTERLEAVED with col stick (1, 2K^2).
    // dst has shape of: BHoWo, C
    // In this file, we do INTERLEAVED reading
    DeviceZoneScopedN("Reader Input Sample Interleaved");
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t dst_addr = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t num_sticks = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_scalar = get_compile_time_arg_val(2);
    constexpr bool src0_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool src1_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr uint32_t src0_stick_nbytes = get_compile_time_arg_val(5);  // all sticks at once
    constexpr uint32_t src0_block_size = get_compile_time_arg_val(6);    // all sticks at once
    constexpr uint32_t src1_stick_nbytes = get_compile_time_arg_val(7);  // per core
    constexpr uint32_t in_height = get_compile_time_arg_val(8);
    constexpr uint32_t in_width = get_compile_time_arg_val(9);
    constexpr uint32_t out_height = get_compile_time_arg_val(10);
    constexpr uint32_t out_width = get_compile_time_arg_val(11);
    constexpr bool align_corners = (bool)get_compile_time_arg_val(12);
    constexpr uint32_t num_input_width_blocks = get_compile_time_arg_val(13);

    uint32_t aligned_src1_unit_size = ((src1_stick_nbytes - 1) & MASK_64) + 128;

    const InterleavedAddrGen<src0_is_dram> s0 = {.bank_base_address = src0_addr, .page_size = src0_stick_nbytes};
    const InterleavedAddrGen<src1_is_dram> s1 = {.bank_base_address = src1_addr, .page_size = src1_stick_nbytes};

    uint32_t ptr1_write_addr = get_write_ptr(cb_id_in1);
    for (uint32_t stick_num = start_id; stick_num < start_id + num_sticks; stick_num++) {
        int32_t b = stick_num / (out_height * out_width);
        uint64_t src_noc_addr = s1.get_noc_addr(stick_num, 0);
        uint32_t ptr1_write_addr = get_write_ptr(cb_id_in1);
        tt::data_movement::common::enhanced_noc_async_read<(src1_stick_nbytes + 128), false>(
            src_noc_addr & MASK_64, ptr1_write_addr, aligned_src1_unit_size);
        ptr1_write_addr = ptr1_write_addr + (src_noc_addr & OFFSET_64);
        volatile tt_l1_ptr uint16_t* src1_stick = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ptr1_write_addr);
        noc_async_read_barrier();

        float x = bfloat16_to_float(src1_stick[0]);
        float y = bfloat16_to_float(src1_stick[1]);

        if (align_corners == 1) {
            x = ((x + 1) * (in_width - 1.f)) / 2.f;
            y = ((y + 1) * (in_height - 1.f)) / 2.f;
        } else {
            x = ((x + 1) * in_width - 1.f) / 2.f;
            y = ((y + 1) * in_height - 1.f) / 2.f;
        }

        cb_reserve_back(cb_id_scalar, 1);
        cb_reserve_back(cb_id_in0, 4);
        uint32_t in_scalar_write_addr = get_write_ptr(cb_id_scalar);
        uint32_t ptr0_write_addr = get_write_ptr(cb_id_in0);
        uint64_t src00_noc_addr = 0, src01_noc_addr = 0, src02_noc_addr = 0, src03_noc_addr = 0;

        volatile tt_l1_ptr uint16_t* scalar = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_scalar_write_addr);
        float w1, w2, w3, w4;
        bool f1 = false, f2 = false, f3 = false, f4 = false;
        if (x <= -1 || y <= -1 || in_height <= y || in_width <= x) {
            w1 = 0.0f;
            w2 = 0.0f;
            w3 = 0.0f;
            w4 = 0.0f;
        } else {
            int16_t x_high = std::ceil(x);
            int16_t x_low = x_high - 1;
            int16_t y_high = std::ceil(y);
            int16_t y_low = y_high - 1;

            float ly = y - (float)y_low;
            float lx = x - (float)x_low;

            if (y_low >= 0 && x_low >= 0) {
                w1 = (1 - ly) * (1 - lx);
                f1 = true;
                src00_noc_addr = s0.get_noc_addr(b * in_height * in_width + y_low * in_width + x_low, 0);
                noc_async_read(src00_noc_addr, ptr0_write_addr, src0_block_size);
            } else {
                w1 = 0.0f;
            }
            ptr0_write_addr += src0_block_size;

            if (y_high >= 0 && y_high <= (int16_t)(in_height - 1) && x_low >= 0) {
                w2 = (1 - lx) * ly;
                f2 = true;
                src01_noc_addr = s0.get_noc_addr(b * in_height * in_width + y_high * in_width + x_low, 0);
                noc_async_read(src01_noc_addr, ptr0_write_addr, src0_block_size);
            } else {
                w2 = 0.0f;
            }
            ptr0_write_addr += src0_block_size;

            if (x_high >= 0 && x_high <= (int16_t)(in_width - 1) && y_low >= 0) {
                w3 = lx * (1 - ly);
                f3 = true;
                src02_noc_addr = s0.get_noc_addr(b * in_height * in_width + y_low * in_width + x_high, 0);
                noc_async_read(src02_noc_addr, ptr0_write_addr, src0_block_size);
            } else {
                w3 = 0.0f;
            }
            ptr0_write_addr += src0_block_size;

            if (x_high >= 0 && x_high <= (int16_t)(in_width - 1) && y_high >= 0 && y_high <= (int16_t)(in_height - 1)) {
                w4 = lx * ly;
                f4 = true;
                src03_noc_addr = s0.get_noc_addr(b * in_height * in_width + y_high * in_width + x_high, 0);
                noc_async_read(src03_noc_addr, ptr0_write_addr, src0_block_size);
            } else {
                w4 = 0.0f;
            }
        }
        scalar[0] = float_to_bfloat16(w1);
        scalar[1] = float_to_bfloat16(w2);
        scalar[2] = float_to_bfloat16(w3);
        scalar[3] = float_to_bfloat16(w4);
        noc_async_read_barrier();

        cb_push_back(cb_id_scalar, 1);
        cb_push_back(cb_id_in0, 4);

        // if stick size is too large, divide into multiple blocks and for the next n-1 blocks, we do not need to read
        // scalar again
        for (uint32_t i = 1; i < num_input_width_blocks; i++) {
            cb_reserve_back(cb_id_scalar, 1);
            cb_reserve_back(cb_id_in0, 4);

            uint32_t ptr0_write_addr = get_write_ptr(cb_id_in0);

            if (f1) {
                noc_async_read(src00_noc_addr + src0_block_size * i, ptr0_write_addr, src0_block_size);
            }
            ptr0_write_addr += src0_block_size;

            if (f2) {
                noc_async_read(src01_noc_addr + src0_block_size * i, ptr0_write_addr, src0_block_size);
            }
            ptr0_write_addr += src0_block_size;

            if (f3) {
                noc_async_read(src02_noc_addr + src0_block_size * i, ptr0_write_addr, src0_block_size);
            }
            ptr0_write_addr += src0_block_size;

            if (f4) {
                noc_async_read(src03_noc_addr + src0_block_size * i, ptr0_write_addr, src0_block_size);
            }
            ptr0_write_addr += src0_block_size;

            noc_async_read_barrier();

            cb_push_back(cb_id_scalar, 1);
            cb_push_back(cb_id_in0, 4);
        }
    }
}
