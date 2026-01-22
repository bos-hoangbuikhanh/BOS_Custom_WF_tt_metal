#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#define ALWI inline __attribute__((always_inline))


ALWI float bfloat16_to_float(uint16_t bfp16_bits) {
    uint32_t float_bits = static_cast<uint32_t>(bfp16_bits) << 16;
    float result;
    std::memcpy(&result, &float_bits, sizeof(result));
    return result;
}

ALWI int16_t bfloat16_to_int16(uint16_t bfp16_bits) {
    const unsigned sign = bfp16_bits >> 15;             // 0 or 1
    const unsigned exp  = (bfp16_bits >> 7) & 0xFFu;    // 0..255
    const unsigned frac = bfp16_bits & 0x7Fu;           // 0..127

    // Zeros & subnormals -> 0 after truncation.
    if (exp == 0u) return 0;

    // Infs/NaNs: typical FP->int converts to INT32_MIN, then narrowing keeps low 16 bits = 0.
    if (exp == 0xFFu) return 0;

    // Normalized: value = (-1)^sign * (128 + frac) * 2^(exp - 134)
    const int  shift = int(exp) - 134;         // 134 = 127 (bias) + 7 (frac bits)
    const unsigned M = 128u + frac;            // 128..255

    // We'll emulate hardware float->int32 with truncation toward zero,
    // then "narrow" to int16 by taking the low 16 bits.
    uint32_t conv32;

    if (shift >= 0) {
        // Detect big shifts without doing UB left-shifts.
        if (shift >= 24) {
            // Special exact case: -2^31 fits in int32.
            // Happens when sign=1, M=128, shift=24.
            if (sign && frac == 0u && shift == 24) {
                conv32 = 0x80000000u;          // INT32_MIN
            } else {
                conv32 = 0x80000000u;          // sentinel for overflow
            }
        } else {
            uint32_t mag = M << unsigned(shift);  // <= 255<<23 fits in int32
            if (sign) {
                // Allow exactly 0x80000000 for INT32_MIN, otherwise sentinel.
                if (mag > 0x80000000u) conv32 = 0x80000000u;
                else if (mag == 0x80000000u) conv32 = 0x80000000u;
                else conv32 = uint32_t(-(int32_t)mag);
            } else {
                conv32 = (mag > 0x7FFFFFFFu) ? 0x80000000u : mag;
            }
        }
    } else {
        // Right shift -> truncation toward zero on the magnitude.
        const unsigned r = unsigned(-shift);
        const uint32_t mag = M >> r;
        conv32 = sign ? uint32_t(-(int32_t)mag) : mag;
    }

    // Narrow like "int32 -> int16" keeping low 16 bits.
    return static_cast<int16_t>(static_cast<uint16_t>(conv32));
}

ALWI uint16_t float_to_bfloat16(float value) {
    uint32_t float_bits;
    std::memcpy(&float_bits, &value, sizeof(float));
    return static_cast<uint16_t>(float_bits >> 16);
}

ALWI uint32_t flatten_idx(uint32_t row, uint32_t col, uint32_t scale = 1) {
    // DeviceZoneScopedN("flatten_idx");
    uint32_t face_h = row >> 4;
    uint32_t face_w = (col * scale) >> 4;
    uint32_t face_offset = (face_h << 1) + face_w;
    uint32_t intra_h = row & 0xF;
    uint32_t intra_w = (col * scale) & 0xF;
    return (face_offset << 8) + (intra_h << 4) + intra_w;
}