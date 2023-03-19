#pragma once
#include <stdint.h>

#include "viterbi_decoder_scalar.h"
#include "viterbi_decoder_sse_u16.h"
#include "viterbi_decoder_avx_u16.h"
#include "viterbi_decoder_sse_u8.h"
#include "viterbi_decoder_avx_u8.h"

// NOTE: Use these classes inside template parameters 
//       so factory code is generated inside the function template
template <size_t K, size_t R>
class ViterbiDecoder_Factory_u16
{
public:
    using Scalar = ViterbiDecoder_Scalar<K,R,uint16_t,int16_t>;
    using SIMD_SSE = ViterbiDecoder_SSE_u16<K,R>;
    using SIMD_AVX = ViterbiDecoder_AVX_u16<K,R>;
};

template <size_t K, size_t R>
class ViterbiDecoder_Factory_u8
{
public:
    using Scalar = ViterbiDecoder_Scalar<K,R,uint8_t,int8_t>;
    using SIMD_SSE = ViterbiDecoder_SSE_u8<K,R>;
    using SIMD_AVX = ViterbiDecoder_AVX_u8<K,R>;
};