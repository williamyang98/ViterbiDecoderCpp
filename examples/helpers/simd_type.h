#pragma once

#if defined(VITERBI_SIMD_X86)
#pragma message("Compiling with x86 vectorisations")
#elif defined(VITERBI_SIMD_ARM)
#pragma message("Compiling with arm vectorisations")
#else
#pragma message("Compiling with no vectorisations")
#endif

enum SIMD_Type {
    SCALAR=0, 
    #if defined(VITERBI_SIMD_X86)
    SIMD_SSE=1, SIMD_AVX=2
    #elif defined(VITERBI_SIMD_ARM)
    SIMD_NEON=1
    #endif
};

template <class decoder_factory_t>
constexpr 
SIMD_Type get_fastest_simd_type() {
    #if defined(VITERBI_SIMD_X86)
    if constexpr(decoder_factory_t::SIMD_AVX::is_valid) {
        return SIMD_Type::SIMD_AVX;
    }  
    if constexpr(decoder_factory_t::SIMD_SSE::is_valid) {
        return SIMD_Type::SIMD_SSE;
    } 
    #elif defined(VITERBI_SIMD_ARM)
    if constexpr(decoder_factory_t::SIMD_NEON::is_valid) {
        return SIMD_Type::SIMD_NEON;
    } 
    #endif

    static_assert(decoder_factory_t::Scalar::is_valid, "Scalar decoder is invalid in decoder factory");
    return SIMD_Type::SCALAR;
};

constexpr 
const char* get_simd_type_string(const SIMD_Type simd_type) {
    switch (simd_type) {
    case SIMD_Type::SCALAR:    return "SCALAR";
    #if defined(VITERBI_SIMD_X86)
    case SIMD_Type::SIMD_SSE:  return "SIMD_SSE";
    case SIMD_Type::SIMD_AVX:  return "SIMD_AVX";
    #elif defined(VITERBI_SIMD_ARM)
    case SIMD_Type::SIMD_NEON: return "SIMD_NEON";
    #endif
    default:                   return "UNKNOWN";
    }
}
