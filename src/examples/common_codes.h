#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string>
#include <array>

// NOTE: We have a bunch of wierd template code since each viterbi decoder
//       instantiates a different template

// Sample codes
template <size_t constraint_length, size_t code_rate, typename code_t = uint16_t>
struct Code {
    static constexpr size_t K = constraint_length;
    static constexpr size_t R = code_rate;
    const char* name;
    std::array<code_t, R> G;
};

// NOTE: The codes are vaguely sorted by complexity (approximated as K*R)
// Source: https://www.spiral.net/software/viterbi.html
struct {
    Code< 3, 2> code_0{ "Basic K=3 R=1/2", { 0b111, 0b101 } };
    Code< 5, 2> code_1{ "Basic K=5 R=1/2", { 0b10111, 0b11001 } };
    Code< 7, 2> code_2{ "Voyager",         { 109, 79} };
    Code< 7, 3> code_3{ "LTE",             { 91, 117, 121 } };
    Code< 7, 4> code_4{ "DAB Radio",       { 109, 79, 83, 109 } }; 
    Code< 9, 2> code_5{ "CDMA IS-95A",     { 491, 369 } };
    Code< 9, 4> code_6{ "CDMA 2000",       { 501, 441, 331, 315 } };
    Code<15, 6> code_7{ "Cassini",         { 17817, 20133, 23879, 30451, 32439, 26975 } };
    const size_t N = 8;
} common_codes;

enum DecodeType {
    SCALAR=0, SIMD_SSE=1, SIMD_AVX=2
};

template <class decoder_factory_t>
constexpr 
DecodeType get_fastest_decode_type() {
    if constexpr(decoder_factory_t::SIMD_AVX::is_valid) {
        return DecodeType::SIMD_AVX;
    } else if constexpr(decoder_factory_t::SIMD_SSE::is_valid) {
        return DecodeType::SIMD_SSE;
    } else if constexpr(decoder_factory_t::Scalar::is_valid) {
        return DecodeType::SCALAR;
    } else {
        // static_assert(false, "No valid decoder in decoder factory");
    }
};

constexpr 
const char* get_decode_type_string(const DecodeType decode_type) {
    switch (decode_type) {
    case DecodeType::SCALAR:   return "SCALAR";
    case DecodeType::SIMD_SSE: return "SIMD_SSE";
    case DecodeType::SIMD_AVX: return "SIMD_AVX";
    default:                   return "UNKNOWN";
    }
}

template <template <size_t, size_t> class decoder_factory_t, size_t K, size_t R, typename code_t>
constexpr
void print_code(const Code<K,R,code_t>& code, const size_t id, const size_t max_name_length) {
    printf("%2zu | %*s | %2zu %2zu | ", id, (int)max_name_length, code.name, K, R);

    // default decode type
    constexpr auto decode_type = get_fastest_decode_type<decoder_factory_t<K,R>>();
    constexpr const char* decode_name = get_decode_type_string(decode_type);
    printf("%*s", 8, decode_name);
    printf(" | ");
    
    // Coefficients in decimal form
    const auto& G = code.G;
    const size_t N = G.size();
    {
        printf("[");
        size_t i = 0u;
        while (true) {
            printf("%u", G[i]);
            i++;
            if (i >= N) {
                break;
            }
            printf(",");
        }
        printf("]");
    }
    printf("\n");
}

template <template <size_t, size_t> class decoder_factory_t>
void list_codes() {
    constexpr size_t max_name_length = 16;
    printf("ID | %*s |  K  R |     Type | Coefficients\n", (int)max_name_length, "Name");
    print_code<decoder_factory_t>(common_codes.code_0, 0, max_name_length);
    print_code<decoder_factory_t>(common_codes.code_1, 1, max_name_length);
    print_code<decoder_factory_t>(common_codes.code_2, 2, max_name_length);
    print_code<decoder_factory_t>(common_codes.code_3, 3, max_name_length);
    print_code<decoder_factory_t>(common_codes.code_4, 4, max_name_length);
    print_code<decoder_factory_t>(common_codes.code_5, 5, max_name_length);
    print_code<decoder_factory_t>(common_codes.code_6, 6, max_name_length);
    print_code<decoder_factory_t>(common_codes.code_7, 7, max_name_length);
}
