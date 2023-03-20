#pragma once
#include <stdint.h>

#include "viterbi_decoder_scalar.h"
#include "simd_type.h"

#if defined(VITERBI_SIMD_X86)
#include "x86/viterbi_decoder_sse_u16.h"
#include "x86/viterbi_decoder_avx_u16.h"
#include "x86/viterbi_decoder_sse_u8.h"
#include "x86/viterbi_decoder_avx_u8.h"
#elif defined(VITERBI_SIMD_ARM)
#include "arm/viterbi_decoder_neon_u8.h"
#include "arm/viterbi_decoder_neon_u16.h"
#endif

// NOTE: Use these classes inside template parameters 
//       so factory code is generated inside the function template
template <size_t K, size_t R>
class ViterbiDecoder_Factory_u16
{
public:
    using Scalar = ViterbiDecoder_Scalar<K,R,uint16_t,int16_t>;
    #if defined(VITERBI_SIMD_X86)
    using SIMD_SSE = ViterbiDecoder_SSE_u16<K,R>;
    using SIMD_AVX = ViterbiDecoder_AVX_u16<K,R>;
    #elif defined(VITERBI_SIMD_ARM)
    using SIMD_NEON = ViterbiDecoder_NEON_u16<K,R>;
    #endif
};

template <size_t K, size_t R>
class ViterbiDecoder_Factory_u8
{
public:
    using Scalar = ViterbiDecoder_Scalar<K,R,uint8_t,int8_t>;
    #if defined(VITERBI_SIMD_X86)
    using SIMD_SSE = ViterbiDecoder_SSE_u8<K,R>;
    using SIMD_AVX = ViterbiDecoder_AVX_u8<K,R>;
    #elif defined(VITERBI_SIMD_ARM)
    using SIMD_NEON = ViterbiDecoder_NEON_u8<K,R>;
    #endif
};
