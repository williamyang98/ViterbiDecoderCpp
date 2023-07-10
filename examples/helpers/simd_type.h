#pragma once

#include <vector>

#include "viterbi/viterbi_decoder_scalar.h"
#include "../arch/simd_flags.h"

#if defined(__SSE4_2__)
#include "viterbi/x86/viterbi_decoder_sse_u16.h"
#include "viterbi/x86/viterbi_decoder_sse_u8.h"
#endif
#if defined(__AVX2__)
#include "viterbi/x86/viterbi_decoder_avx_u16.h"
#include "viterbi/x86/viterbi_decoder_avx_u8.h"
#endif
#if defined(__SIMD_NEON__)
#include "viterbi/arm/viterbi_decoder_neon_u8.h"
#include "viterbi/arm/viterbi_decoder_neon_u16.h"
#endif

enum SIMD_Type {
    SCALAR=0, 
    #if defined(__SSE4_2__)
    SIMD_SSE=1, 
    #endif
    #if defined(__AVX2__)
    SIMD_AVX=2,
    #endif
    #if defined(__SIMD_NEON__)
    SIMD_NEON=3,
    #endif
};


const std::vector<SIMD_Type> SIMD_Type_List = {
    SIMD_Type::SCALAR,
    #if defined(__SSE4_2__)
    SIMD_Type::SIMD_SSE, 
    #endif
    #if defined(__AVX2__)
    SIMD_Type::SIMD_AVX,
    #endif
    #if defined(__SIMD_NEON__)
    SIMD_Type::SIMD_NEON,
    #endif
};

// NOTE: Use these classes inside template parameters 
//       so factory code is generated inside the function template
class ViterbiDecoder_Factory_u16
{
public:
    template <size_t K, size_t R>
    using SCALAR = ViterbiDecoder_Scalar<K,R,uint16_t,int16_t>;
    #if defined(__SSE4_2__)
    template <size_t K, size_t R>
    using SIMD_SSE = ViterbiDecoder_SSE_u16<K,R>;
    #endif
    #if defined(__AVX2__)
    template <size_t K, size_t R>
    using SIMD_AVX = ViterbiDecoder_AVX_u16<K,R>;
    #endif
    #if defined(__SIMD_NEON__)
    template <size_t K, size_t R>
    using SIMD_NEON = ViterbiDecoder_NEON_u16<K,R>;
    #endif
};

class ViterbiDecoder_Factory_u8
{
public:
    template <size_t K, size_t R>
    using SCALAR = ViterbiDecoder_Scalar<K,R,uint8_t,int8_t>;
    #if defined(__SSE4_2__)
    template <size_t K, size_t R>
    using SIMD_SSE = ViterbiDecoder_SSE_u8<K,R>;
    #endif
    #if defined(__AVX2__)
    template <size_t K, size_t R>
    using SIMD_AVX = ViterbiDecoder_AVX_u8<K,R>;
    #endif
    #if defined(__SIMD_NEON__)
    template <size_t K, size_t R>
    using SIMD_NEON = ViterbiDecoder_NEON_u8<K,R>;
    #endif
};

#if defined(__SSE4_2__)
#define __SELECT_FACTORY_ITEM_SSE(FACTORY, INDEX, K, R, BLOCK) case SIMD_Type::SIMD_SSE: { using it = typename FACTORY::SIMD_SSE<K,R>; BLOCK }; break;
#else
#define __SELECT_FACTORY_ITEM_SSE(FACTORY, INDEX, K, R, BLOCK)
#endif
#if defined(__AVX2__)
#define __SELECT_FACTORY_ITEM_AVX(FACTORY, INDEX, K, R, BLOCK) case SIMD_Type::SIMD_AVX: { using it = typename FACTORY::SIMD_AVX<K,R>; BLOCK }; break;
#else
#define __SELECT_FACTORY_ITEM_AVX(FACTORY, INDEX, K, R, BLOCK)
#endif
#if defined(__SIMD_NEON__)
#define __SELECT_FACTORY_ITEM_NEON(FACTORY, INDEX, K, R, BLOCK) case SIMD_Type::SIMD_NEON: { using it = typename FACTORY::SIMD_NEON<K,R>; BLOCK }; break;
#else
#define __SELECT_FACTORY_ITEM_NEON(FACTORY, INDEX, K, R, BLOCK)
#endif

#define SELECT_FACTORY_ITEM(FACTORY, INDEX, K, R, BLOCK) do {\
    switch (INDEX) {\
    case SIMD_Type::SCALAR:   { using it = typename FACTORY::SCALAR<K,R>;   BLOCK }; break;\
    __SELECT_FACTORY_ITEM_SSE(FACTORY, INDEX, K, R, BLOCK)\
    __SELECT_FACTORY_ITEM_AVX(FACTORY, INDEX, K, R, BLOCK)\
    __SELECT_FACTORY_ITEM_NEON(FACTORY, INDEX, K, R, BLOCK)\
    default: break;\
    }\
} while(0)

constexpr 
const char* get_simd_type_string(const SIMD_Type simd_type) {
    switch (simd_type) {
    case SIMD_Type::SCALAR:    return "SCALAR";
    #if defined(__SSE4_2__)
    case SIMD_Type::SIMD_SSE:  return "SIMD_SSE";
    #endif
    #if defined(__AVX2__)
    case SIMD_Type::SIMD_AVX:  return "SIMD_AVX";
    #endif
    #if defined(__SIMD_NEON__)
    case SIMD_Type::SIMD_NEON: return "SIMD_NEON";
    #endif
    default:                   return "UNKNOWN";
    }
}

template <class factory_t, size_t K, size_t R>
SIMD_Type get_fastest_simd_type() {
    SIMD_Type fastest_type = SIMD_Type::SCALAR;
    for (const auto& simd_type: SIMD_Type_List) {
        SELECT_FACTORY_ITEM(factory_t, simd_type, K, R, { 
            using decoder_t = it;
            if constexpr(!decoder_t::is_valid) { 
                return fastest_type; 
            } 
            fastest_type = simd_type; 
        });
    }
    return fastest_type;
};