#pragma once
#include <stdint.h>

#include "viterbi/viterbi_decoder.h"
#include "viterbi/viterbi_decoder_scalar.h"
#include "viterbi/viterbi_decoder_sse_u16.h"
#include "viterbi/viterbi_decoder_avx_u16.h"
#include "viterbi/viterbi_decoder_sse_u8.h"
#include "viterbi/viterbi_decoder_avx_u8.h"

// NOTE: Use these classes inside template parameters 
//       so factory code is generated inside the function template

class ViterbiDecoder_Factory_u16
{
public:
    template <typename ... U>
    static
    ViterbiDecoder<int16_t, uint64_t>* get_scalar(U&& ... args) {
        return new ViterbiDecoder_Scalar<uint16_t,int16_t>(std::forward<U>(args)...);
    }

    template <typename ... U>
    static
    ViterbiDecoder<int16_t, uint64_t>* get_simd_sse(U&& ... args) {
        return new ViterbiDecoder_SSE_u16<uint64_t>(std::forward<U>(args)...);
    }

    template <typename ... U>
    static
    ViterbiDecoder<int16_t, uint64_t>* get_simd_avx(U&& ... args) {
        return new ViterbiDecoder_AVX_u16<uint64_t>(std::forward<U>(args)...);
    }

    static constexpr 
    size_t K_simd_requirements[3] = {
        0u,
        ViterbiDecoder_SSE_u16<uint64_t>::K_min,
        ViterbiDecoder_AVX_u16<uint64_t>::K_min
    };
};

class ViterbiDecoder_Factory_u8
{
public:
    template <typename ... U>
    static
    ViterbiDecoder<int8_t, uint64_t>* get_scalar(U&& ... args) {
        return new ViterbiDecoder_Scalar<uint8_t,int8_t>(std::forward<U>(args)...);
    }

    template <typename ... U>
    static
    ViterbiDecoder<int8_t, uint64_t>* get_simd_sse(U&& ... args) {
        return new ViterbiDecoder_SSE_u8<uint64_t>(std::forward<U>(args)...);
    }

    template <typename ... U>
    static
    ViterbiDecoder<int8_t, uint64_t>* get_simd_avx(U&& ... args) {
        return new ViterbiDecoder_AVX_u8<uint64_t>(std::forward<U>(args)...);
    }

    static constexpr 
    size_t K_simd_requirements[3] = {
        0u,
        ViterbiDecoder_SSE_u8<uint64_t>::K_min,
        ViterbiDecoder_AVX_u8<uint64_t>::K_min
    };
};