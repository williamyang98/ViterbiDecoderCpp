#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

// Sample codes
enum DecodeType { 
    SCALAR=0, SIMD_SSE=1, SIMD_AVX=2
};

struct Code {
    const std::string name;
    const size_t K;
    const size_t R;
    const std::vector<uint16_t> G;
};

// NOTE: The codes are vaguely sorted by complexity (approximated as K*R)
// Source: https://www.spiral.net/software/viterbi.html
const auto common_codes = std::vector<Code>{
    { "Basic K=3 R=1/2",  3, 2, { 0b111, 0b101 } },
    { "Basic K=5 R=1/2",  5, 2, { 0b10111, 0b11001 } },
    { "Voyager",          7, 2, { 109, 79} },
    { "LTE",              7, 3, { 91, 117, 121 } },
    { "DAB Radio",        7, 4, { 109, 79, 83, 109 } }, 
    { "CDMA IS-95A",      9, 2, { 491, 369 } },
    { "CDMA 2000",        9, 4, { 501, 441, 331, 315 } },
    { "Cassini",         15, 6, { 17817, 20133, 23879, 30451, 32439, 26975 } }
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


static
void list_codes(const Code* codes, const size_t N, const size_t K_simd[3]) {
    size_t max_name_length = 0u;
    for (size_t i = 0u; i < N; i++) {
        const auto& code = codes[i];
        const size_t len = code.name.length();
        max_name_length = std::max(max_name_length, len);
    }

    printf("ID | %*s |  K  R |     Type | Coefficients\n", (int)max_name_length, "Name");

    for (size_t i = 0u; i < N; i++) {
        const auto& code = codes[i];
        const auto decode_type = get_fastest_simd_type(code.K, K_simd);
        printf("%+2zu | %*s | %+2zu %+2zu | ", i, (int)max_name_length, code.name.c_str(), code.K, code.R);

        // default decode type
        printf("%*s", 8, get_decode_type_name(decode_type).c_str());
        printf(" | ");
        
        // Coefficients in decimal form
        const auto& G = code.G;
        const size_t M = G.size();
        {
            printf("[");
            size_t j = 0u;
            while (true) {
                printf("%u", G[j]);
                j++;
                if (j >= M) {
                    break;
                }
                printf(",");
            }
            printf("]");
        }

        printf("\n");
    }
}
