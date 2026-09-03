#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// Exact CUDA-inline-PTX port of flash-attention's cute.utils.e2e_asm2.
__device__ __forceinline__ float2 fp4pv_ex2_alu_emulation_f32x2(float2 xy) {
    uint32_t out_x;
    uint32_t out_y;
    asm(
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "}\n"
        : "=r"(out_x), "=r"(out_y)
        : "f"(xy.x), "f"(xy.y));
    return {__uint_as_float(out_x), __uint_as_float(out_y)};
}

// Degree-2 variant for latency/accuracy sweeps. It retains the same packed
// range reduction and exponent reconstruction while removing one packed FMA.
__device__ __forceinline__ float2
fp4pv_ex2_alu_emulation_degree2_f32x2(float2 xy) {
    uint32_t out_x;
    uint32_t out_y;
    asm(
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f6, 0f3EA903CA;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F2A70E4;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l5, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "}\n"
        : "=r"(out_x), "=r"(out_y)
        : "f"(xy.x), "f"(xy.y));
    return {__uint_as_float(out_x), __uint_as_float(out_y)};
}

// Degree-1 is intentionally paired with the coarse E2M1 conversion. Its
// precision is not suitable as a general exp2 replacement, but many of its
// errors disappear after FP4 rounding.
__device__ __forceinline__ float2
fp4pv_ex2_alu_emulation_degree1_f32x2(float2 xy) {
    uint32_t out_x;
    uint32_t out_y;
    asm(
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5;\n\t"
        ".reg .b64 l1, l2, l3, l4, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f5, 0f3F317218;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l4, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "}\n"
        : "=r"(out_x), "=r"(out_y)
        : "f"(xy.x), "f"(xy.y));
    return {__uint_as_float(out_x), __uint_as_float(out_y)};
}

__device__ __forceinline__ float2 fp4pv_ex2_native_f32x2(float2 xy) {
    float2 out;
    asm(
        "ex2.approx.ftz.f32 %0, %2;\n\t"
        "ex2.approx.ftz.f32 %1, %3;\n"
        : "=f"(out.x), "=f"(out.y)
        : "f"(xy.x), "f"(xy.y));
    return out;
}

__device__ __forceinline__ uint32_t fp4pv_pack_e2m1_pair(float2 xy) {
    uint32_t out;
    asm(
        "{\n\t"
        ".reg .b8 packed;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 packed, %2, %1;\n\t"
        "mov.b32 %0, {packed, 0, 0, 0};\n\t"
        "}\n"
        : "=r"(out)
        : "f"(xy.x), "f"(xy.y));
    return out;
}

template <int Begin, int End, typename Function>
__device__ __forceinline__ void fp4pv_stage2_static_for(Function &&function) {
    if constexpr (Begin < End) {
        function(std::integral_constant<int, Begin>{});
        fp4pv_stage2_static_for<Begin + 1, End>(static_cast<Function &&>(function));
    }
}

template <int Period, uint32_t Mask, int PairIndex>
__device__ __forceinline__ float2 fp4pv_stage2_exp2_pair(float2 xy) {
    static_assert(Period == -1 || (Period >= 1 && Period <= 16),
        "Stage2 EX2 cadence must be all-emulated or use a period up to 16 pairs");
    constexpr bool emulate = [] {
        if constexpr (Period == -1) {
            return true;
        } else {
            return (Mask & (uint32_t{1} << (PairIndex % Period))) != 0;
        }
    }();
    if constexpr (emulate) {
#if TK_HAO_DIRECT_FP4PV_EX2_ALU_DEGREE == 1
        return fp4pv_ex2_alu_emulation_degree1_f32x2(xy);
#elif TK_HAO_DIRECT_FP4PV_EX2_ALU_DEGREE == 2
        return fp4pv_ex2_alu_emulation_degree2_f32x2(xy);
#else
        return fp4pv_ex2_alu_emulation_f32x2(xy);
#endif
    } else {
        return fp4pv_ex2_native_f32x2(xy);
    }
}

struct fp4pv_ex2_alu_detail_f32x2 {
    float2 value;
    float2 fraction;
    float2 polynomial;
    int2 floor_value;
};

struct fp4pv_ex2_range_f32x2 {
    float2 fraction;
    int2 floor_value;
};

// The addendum paths need the range-reduction intermediates. Keep this
// separate from the retained e16 helper above so its generated code is stable.
__device__ __forceinline__ fp4pv_ex2_alu_detail_f32x2
fp4pv_ex2_alu_detail_degree3_f32x2(float2 xy) {
    uint32_t out_x;
    uint32_t out_y;
    uint32_t rounded_x;
    uint32_t rounded_y;
    uint32_t frac_x;
    uint32_t frac_y;
    uint32_t poly_x;
    uint32_t poly_y;
    asm(
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;\n\t"
        "max.ftz.f32 f1, %8, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %9, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "mov.b64 {r9, r10}, l9;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "mov.b32 %2, r1;\n\t"
        "mov.b32 %3, r2;\n\t"
        "mov.b32 %4, r9;\n\t"
        "mov.b32 %5, r10;\n\t"
        "mov.b32 %6, r3;\n\t"
        "mov.b32 %7, r4;\n\t"
        "}\n"
        : "=r"(out_x), "=r"(out_y), "=r"(rounded_x), "=r"(rounded_y),
          "=r"(frac_x), "=r"(frac_y), "=r"(poly_x), "=r"(poly_y)
        : "f"(xy.x), "f"(xy.y));
    return {
        {__uint_as_float(out_x), __uint_as_float(out_y)},
        {__uint_as_float(frac_x), __uint_as_float(frac_y)},
        {__uint_as_float(poly_x), __uint_as_float(poly_y)},
        {
            static_cast<int>(static_cast<int8_t>(rounded_x & 0xffu)),
            static_cast<int>(static_cast<int8_t>(rounded_y & 0xffu)),
        },
    };
}

__device__ __forceinline__ fp4pv_ex2_alu_detail_f32x2
fp4pv_ex2_alu_detail_degree2_f32x2(float2 xy) {
    uint32_t out_x;
    uint32_t out_y;
    uint32_t rounded_x;
    uint32_t rounded_y;
    uint32_t frac_x;
    uint32_t frac_y;
    uint32_t poly_x;
    uint32_t poly_y;
    asm(
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;\n\t"
        "max.ftz.f32 f1, %8, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %9, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f6, 0f3EA903CA;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F2A70E4;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l5, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "mov.b64 {r9, r10}, l9;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "mov.b32 %2, r1;\n\t"
        "mov.b32 %3, r2;\n\t"
        "mov.b32 %4, r9;\n\t"
        "mov.b32 %5, r10;\n\t"
        "mov.b32 %6, r3;\n\t"
        "mov.b32 %7, r4;\n\t"
        "}\n"
        : "=r"(out_x), "=r"(out_y), "=r"(rounded_x), "=r"(rounded_y),
          "=r"(frac_x), "=r"(frac_y), "=r"(poly_x), "=r"(poly_y)
        : "f"(xy.x), "f"(xy.y));
    return {
        {__uint_as_float(out_x), __uint_as_float(out_y)},
        {__uint_as_float(frac_x), __uint_as_float(frac_y)},
        {__uint_as_float(poly_x), __uint_as_float(poly_y)},
        {
            static_cast<int>(static_cast<int8_t>(rounded_x & 0xffu)),
            static_cast<int>(static_cast<int8_t>(rounded_y & 0xffu)),
        },
    };
}

__device__ __forceinline__ fp4pv_ex2_range_f32x2
fp4pv_ex2_alu_range_f32x2(float2 xy) {
    uint32_t rounded_x;
    uint32_t rounded_y;
    uint32_t frac_x;
    uint32_t frac_y;
    asm(
        "{\n\t"
        ".reg .f32 f1, f2, f3;\n\t"
        ".reg .b64 l1, l2, l7, l8, l9;\n\t"
        ".reg .s32 r1, r2, r3, r4;\n\t"
        "max.ftz.f32 f1, %4, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %5, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l9;\n\t"
        "mov.b32 %0, r1;\n\t"
        "mov.b32 %1, r2;\n\t"
        "mov.b32 %2, r3;\n\t"
        "mov.b32 %3, r4;\n\t"
        "}\n"
        : "=r"(rounded_x), "=r"(rounded_y), "=r"(frac_x), "=r"(frac_y)
        : "f"(xy.x), "f"(xy.y));
    return {
        {__uint_as_float(frac_x), __uint_as_float(frac_y)},
        {
            static_cast<int>(static_cast<int8_t>(rounded_x & 0xffu)),
            static_cast<int>(static_cast<int8_t>(rounded_y & 0xffu)),
        },
    };
}

__device__ __forceinline__ uint32_t
fp4pv_e2m1_positive_code_from_polynomial(int floor_value, float polynomial) {
    uint32_t code;
    asm(
        "{\n\t"
        ".reg .pred p0, p1, p2, p3, p4, p5;\n\t"
        ".reg .s32 clamped, base, increment0, increment1, temp;\n\t"
        ".reg .f32 threshold0, threshold1;\n\t"
        "max.s32 clamped, %1, -1;\n\t"
        "min.s32 clamped, clamped, 2;\n\t"
        "mul.lo.s32 base, clamped, 2;\n\t"
        "add.s32 base, base, 2;\n\t"
        "setp.eq.s32 p0, clamped, -1;\n\t"
        "selp.s32 temp, 1, 0, p0;\n\t"
        "add.s32 base, base, temp;\n\t"
        "setp.le.s32 p0, %1, -2;\n\t"
        "selp.s32 base, 0, base, p0;\n\t"
        "setp.gt.s32 p1, %1, 2;\n\t"
        "selp.s32 base, 7, base, p1;\n\t"
        "mov.f32 threshold0, 0f3FA00000;\n\t"
        "setp.eq.s32 p2, %1, -1;\n\t"
        "selp.f32 threshold0, 0f3FBFFFFF, threshold0, p2;\n\t"
        "setp.eq.s32 p3, %1, -2;\n\t"
        "selp.f32 threshold0, 0f3F800000, threshold0, p3;\n\t"
        "setp.lt.s32 p4, %1, -2;\n\t"
        "or.pred p4, p4, p1;\n\t"
        "selp.f32 threshold0, 0f7F800000, threshold0, p4;\n\t"
        "setp.eq.s32 p4, %1, 0;\n\t"
        "setp.eq.s32 p5, %1, 1;\n\t"
        "or.pred p4, p4, p5;\n\t"
        "selp.f32 threshold1, 0f3FDFFFFF, 0f7F800000, p4;\n\t"
        "setp.gt.f32 p4, %2, threshold0;\n\t"
        "setp.gt.f32 p5, %2, threshold1;\n\t"
        "selp.s32 increment0, 1, 0, p4;\n\t"
        "selp.s32 increment1, 1, 0, p5;\n\t"
        "add.s32 base, base, increment0;\n\t"
        "add.s32 %0, base, increment1;\n\t"
        "}\n"
        : "=r"(code)
        : "r"(floor_value), "f"(polynomial));
    return code;
}

__device__ __forceinline__ uint32_t
fp4pv_e2m1_positive_code_from_log_fraction(int floor_value, float fraction) {
    uint32_t code;
    asm(
        "{\n\t"
        ".reg .pred p0, p1, p2, p3, p4, p5;\n\t"
        ".reg .s32 clamped, base, increment0, increment1, temp;\n\t"
        ".reg .f32 threshold0, threshold1;\n\t"
        "max.s32 clamped, %1, -1;\n\t"
        "min.s32 clamped, clamped, 2;\n\t"
        "mul.lo.s32 base, clamped, 2;\n\t"
        "add.s32 base, base, 2;\n\t"
        "setp.eq.s32 p0, clamped, -1;\n\t"
        "selp.s32 temp, 1, 0, p0;\n\t"
        "add.s32 base, base, temp;\n\t"
        "setp.le.s32 p0, %1, -2;\n\t"
        "selp.s32 base, 0, base, p0;\n\t"
        "setp.gt.s32 p1, %1, 2;\n\t"
        "selp.s32 base, 7, base, p1;\n\t"
        "mov.f32 threshold0, 0f3EA4D3C2;\n\t"
        "setp.eq.s32 p2, %1, -1;\n\t"
        "selp.f32 threshold0, 0f3F15C019, threshold0, p2;\n\t"
        "setp.eq.s32 p3, %1, -2;\n\t"
        "selp.f32 threshold0, 0f00000000, threshold0, p3;\n\t"
        "setp.lt.s32 p4, %1, -2;\n\t"
        "or.pred p4, p4, p1;\n\t"
        "selp.f32 threshold0, 0f7F800000, threshold0, p4;\n\t"
        "setp.eq.s32 p4, %1, 0;\n\t"
        "setp.eq.s32 p5, %1, 1;\n\t"
        "or.pred p4, p4, p5;\n\t"
        "selp.f32 threshold1, 0f3F4EAECF, 0f7F800000, p4;\n\t"
        "setp.gt.f32 p4, %2, threshold0;\n\t"
        "setp.gt.f32 p5, %2, threshold1;\n\t"
        "selp.s32 increment0, 1, 0, p4;\n\t"
        "selp.s32 increment1, 1, 0, p5;\n\t"
        "add.s32 base, base, increment0;\n\t"
        "add.s32 %0, base, increment1;\n\t"
        "}\n"
        : "=r"(code)
        : "r"(floor_value), "f"(fraction));
    return code;
}

__device__ __forceinline__ uint32_t
fp4pv_pack_e2m1_pair_from_polynomial(const fp4pv_ex2_alu_detail_f32x2 &detail) {
    const uint32_t x = fp4pv_e2m1_positive_code_from_polynomial(
        detail.floor_value.x, detail.polynomial.x);
    const uint32_t y = fp4pv_e2m1_positive_code_from_polynomial(
        detail.floor_value.y, detail.polynomial.y);
    return x | (y << 4);
}

__device__ __forceinline__ uint32_t
fp4pv_pack_e2m1_pair_from_log(const fp4pv_ex2_range_f32x2 &range) {
    const uint32_t x = fp4pv_e2m1_positive_code_from_log_fraction(
        range.floor_value.x, range.fraction.x);
    const uint32_t y = fp4pv_e2m1_positive_code_from_log_fraction(
        range.floor_value.y, range.fraction.y);
    return x | (y << 4);
}

__device__ __forceinline__ uint32_t
fp4pv_pack_e2m1_pair_from_log(const fp4pv_ex2_alu_detail_f32x2 &detail) {
    const fp4pv_ex2_range_f32x2 range = {detail.fraction, detail.floor_value};
    return fp4pv_pack_e2m1_pair_from_log(range);
}

struct fp4pv_stage2_fp4_native_pair_result {
    float2 value;
    uint32_t packed_byte;
};

// F2 degree-3 fused form: only the reconstructed pair and payload byte cross
// the inline-PTX boundary, allowing ptxas to recycle all range/classifier state.
__device__ __forceinline__ fp4pv_stage2_fp4_native_pair_result
fp4pv_ex2_alu_f2_degree3_pair(float2 xy) {
    uint32_t out_x;
    uint32_t out_y;
    uint32_t packed_byte;
    asm(
        "{\n\t"
        ".reg .pred p0, p1;\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;\n\t"
        "max.ftz.f32 f1, %3, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %4, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "mov.b64 {r9, r10}, l9;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "bfe.s32 r1, r1, 0, 8;\n\t"
        "bfe.s32 r2, r2, 0, 8;\n\t"

        // X lane: base code and first region-dependent threshold.
        "max.s32 r7, r1, -1;\n\t"
        "min.s32 r7, r7, 2;\n\t"
        "mul.lo.s32 r7, r7, 2;\n\t"
        "add.s32 r7, r7, 2;\n\t"
        "setp.eq.s32 p0, r1, -1;\n\t"
        "selp.s32 r6, 1, 0, p0;\n\t"
        "add.s32 r7, r7, r6;\n\t"
        "setp.le.s32 p0, r1, -2;\n\t"
        "selp.s32 r7, 0, r7, p0;\n\t"
        "setp.gt.s32 p1, r1, 2;\n\t"
        "selp.s32 r7, 7, r7, p1;\n\t"
        "mov.f32 f1, 0f3EA4D3C2;\n\t"
        "setp.eq.s32 p0, r1, -1;\n\t"
        "selp.f32 f1, 0f3F15C019, f1, p0;\n\t"
        "setp.eq.s32 p0, r1, -2;\n\t"
        "selp.f32 f1, 0f00000000, f1, p0;\n\t"
        "setp.lt.s32 p0, r1, -2;\n\t"
        "or.pred p0, p0, p1;\n\t"
        "selp.f32 f1, 0f7F800000, f1, p0;\n\t"
        "mov.b32 f2, r9;\n\t"
        "setp.gt.f32 p0, f2, f1;\n\t"
        "selp.s32 r6, 1, 0, p0;\n\t"
        "add.s32 r7, r7, r6;\n\t"
        "setp.eq.s32 p0, r1, 0;\n\t"
        "setp.eq.s32 p1, r1, 1;\n\t"
        "or.pred p0, p0, p1;\n\t"
        "selp.f32 f1, 0f3F4EAECF, 0f7F800000, p0;\n\t"
        "setp.gt.f32 p0, f2, f1;\n\t"
        "selp.s32 r6, 1, 0, p0;\n\t"
        "add.s32 r7, r7, r6;\n\t"

        // Y lane reuses all temporary registers while preserving X in r7.
        "max.s32 r8, r2, -1;\n\t"
        "min.s32 r8, r8, 2;\n\t"
        "mul.lo.s32 r8, r8, 2;\n\t"
        "add.s32 r8, r8, 2;\n\t"
        "setp.eq.s32 p0, r2, -1;\n\t"
        "selp.s32 r6, 1, 0, p0;\n\t"
        "add.s32 r8, r8, r6;\n\t"
        "setp.le.s32 p0, r2, -2;\n\t"
        "selp.s32 r8, 0, r8, p0;\n\t"
        "setp.gt.s32 p1, r2, 2;\n\t"
        "selp.s32 r8, 7, r8, p1;\n\t"
        "mov.f32 f1, 0f3EA4D3C2;\n\t"
        "setp.eq.s32 p0, r2, -1;\n\t"
        "selp.f32 f1, 0f3F15C019, f1, p0;\n\t"
        "setp.eq.s32 p0, r2, -2;\n\t"
        "selp.f32 f1, 0f00000000, f1, p0;\n\t"
        "setp.lt.s32 p0, r2, -2;\n\t"
        "or.pred p0, p0, p1;\n\t"
        "selp.f32 f1, 0f7F800000, f1, p0;\n\t"
        "mov.b32 f2, r10;\n\t"
        "setp.gt.f32 p0, f2, f1;\n\t"
        "selp.s32 r6, 1, 0, p0;\n\t"
        "add.s32 r8, r8, r6;\n\t"
        "setp.eq.s32 p0, r2, 0;\n\t"
        "setp.eq.s32 p1, r2, 1;\n\t"
        "or.pred p0, p0, p1;\n\t"
        "selp.f32 f1, 0f3F4EAECF, 0f7F800000, p0;\n\t"
        "setp.gt.f32 p0, f2, f1;\n\t"
        "selp.s32 r6, 1, 0, p0;\n\t"
        "add.s32 r8, r8, r6;\n\t"
        "shl.b32 r8, r8, 4;\n\t"
        "or.b32 r7, r7, r8;\n\t"
        "mov.b32 %2, r7;\n\t"
        "}\n"
        : "=r"(out_x), "=r"(out_y), "=r"(packed_byte)
        : "f"(xy.x), "f"(xy.y));
    return {{__uint_as_float(out_x), __uint_as_float(out_y)}, packed_byte};
}

// Sum 2x the nonnegative E2M1 values in one packed x8 payload word.
__device__ __forceinline__ int
fp4pv_stage2_qsum_weights_from_word(uint32_t word, int acc) {
    const uint32_t lo = word & 0x07070707u;
    const uint32_t hi = (word >> 4) & 0x07070707u;
    const uint32_t sel_lo =
        (lo & 0x0fu) | ((lo >> 4) & 0x00f0u) |
        ((lo >> 8) & 0x0f00u) | ((lo >> 12) & 0xf000u);
    const uint32_t sel_hi =
        (hi & 0x0fu) | ((hi >> 4) & 0x00f0u) |
        ((hi >> 8) & 0x0f00u) | ((hi >> 12) & 0xf000u);
    uint32_t weights_lo;
    uint32_t weights_hi;
    asm("prmt.b32 %0, %1, %2, %3;"
        : "=r"(weights_lo)
        : "r"(0x03020100u), "r"(0x0c080604u), "r"(sel_lo));
    asm("prmt.b32 %0, %1, %2, %3;"
        : "=r"(weights_hi)
        : "r"(0x03020100u), "r"(0x0c080604u), "r"(sel_hi));
    asm("dp4a.s32.s32 %0, %1, %2, %0;"
        : "+r"(acc)
        : "r"(weights_lo), "r"(0x01010101u));
    asm("dp4a.s32.s32 %0, %1, %2, %0;"
        : "+r"(acc)
        : "r"(weights_hi), "r"(0x01010101u));
    return acc;
}
