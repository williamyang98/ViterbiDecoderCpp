#pragma once
#include <stddef.h>
#include <string>

enum DecodeType { 
    SCALAR=0, SIMD_SSE=1, SIMD_AVX=2
};

static 
const std::string get_decode_type_name(DecodeType d) {
    switch (d) {
    case DecodeType::SCALAR:   return "Scalar";
    case DecodeType::SIMD_SSE: return "SIMD_SSE";
    case DecodeType::SIMD_AVX: return "SIMD_AVX";
    default:                   return "UNKNOWN";
    }
}

// K_simd contains the minimum constraint lengths required for:
// 1. Scalar code
// 2. SSE code
// 3. AVX code
static
DecodeType get_fastest_simd_type(const size_t K, const size_t K_simd[3]) {
    size_t simd_type = 0u;
    for (size_t i = 0u; i < 3u; i++) {
        if (K < K_simd[i]) {
            break;
        }
        simd_type = i;
    }
    return DecodeType(simd_type);
};