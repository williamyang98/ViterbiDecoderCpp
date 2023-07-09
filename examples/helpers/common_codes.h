#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string>
#include <array>
#include "simd_type.h"

// Sample codes
template <size_t constraint_length, size_t code_rate, typename code_t>
struct Code {
    static constexpr size_t K = constraint_length;
    static constexpr size_t R = code_rate;
    const char* name;
    std::array<code_t, R> G;
};

// NOTE: The codes are vaguely sorted by complexity (approximated as K*R)
// Source: https://www.spiral.net/software/viterbi.html
struct {
    Code< 3, 2, uint8_t > code_0{ "Basic K=3 R=1/2", { 0b111, 0b101 } };
    Code< 5, 2, uint8_t > code_1{ "Basic K=5 R=1/2", { 0b10111, 0b11001 } };
    Code< 7, 2, uint8_t > code_2{ "Voyager",         { 109, 79} };
    Code< 7, 3, uint8_t > code_3{ "LTE",             { 91, 117, 121 } };
    Code< 7, 4, uint8_t > code_4{ "DAB Radio",       { 109, 79, 83, 109 } }; 
    Code< 9, 2, uint16_t> code_5{ "CDMA IS-95A",     { 491, 369 } };
    Code< 9, 4, uint16_t> code_6{ "CDMA 2000",       { 501, 441, 331, 315 } };
    Code<15, 6, uint16_t> code_7{ "Cassini",         { 17817, 20133, 23879, 30451, 32439, 26975 } };
    const size_t N = 8;
} COMMON_CODES;

#define FOR_COMMON_CODES(BLOCK) do {\
    { const auto& it = COMMON_CODES.code_0; BLOCK }\
    { const auto& it = COMMON_CODES.code_1; BLOCK }\
    { const auto& it = COMMON_CODES.code_2; BLOCK }\
    { const auto& it = COMMON_CODES.code_3; BLOCK }\
    { const auto& it = COMMON_CODES.code_4; BLOCK }\
    { const auto& it = COMMON_CODES.code_5; BLOCK }\
    { const auto& it = COMMON_CODES.code_6; BLOCK }\
    { const auto& it = COMMON_CODES.code_7; BLOCK }\
} while(0)

#define SELECT_COMMON_CODES(INDEX, BLOCK) do {\
    switch (INDEX) {\
    case 0: { const auto& it = COMMON_CODES.code_0; BLOCK }; break;\
    case 1: { const auto& it = COMMON_CODES.code_1; BLOCK }; break;\
    case 2: { const auto& it = COMMON_CODES.code_2; BLOCK }; break;\
    case 3: { const auto& it = COMMON_CODES.code_3; BLOCK }; break;\
    case 4: { const auto& it = COMMON_CODES.code_4; BLOCK }; break;\
    case 5: { const auto& it = COMMON_CODES.code_5; BLOCK }; break;\
    case 6: { const auto& it = COMMON_CODES.code_6; BLOCK }; break;\
    case 7: { const auto& it = COMMON_CODES.code_7; BLOCK }; break;\
    default: break;\
    }\
} while(0)

template <class factory_t, size_t K, size_t R, typename code_t>
void print_code(const Code<K,R,code_t>& code, const size_t id, const size_t max_name_length) {
    printf("%2zu | %*s | %2zu %2zu | ", id, int(max_name_length), code.name, K, R);

    // default decode type
    SIMD_Type simd_type = get_fastest_simd_type<factory_t, K, R>();
    const char* decode_name = get_simd_type_string(simd_type);
    printf("%*s", 9, decode_name);
    printf(" | ");
    
    // Coefficients in decimal form
    const auto& G = code.G;
    const size_t N = G.size();
    printf("[");
    for (size_t i = 0u; i < N; i++) {
        printf("%u", G[i]);
        if (i != (N-1)) {
            printf(",");
        }
    }
    printf("]");
    printf("\n");
}

template <class factory_t>
void list_codes() {
    constexpr size_t max_name_length = 16;
    printf("ID | %*s |  K  R |     Type | Coefficients\n", int(max_name_length), "Name");
    size_t index = 0;
    FOR_COMMON_CODES({
        print_code<factory_t>(it, index, max_name_length);
        index++;
    });
}
