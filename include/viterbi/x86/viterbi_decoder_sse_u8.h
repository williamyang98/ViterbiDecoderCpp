/* Generic Viterbi decoder,
 * Copyright Phil Karn, KA9Q,
 * Karn's original code can be found here: https://github.com/ka9q/libfec 
 * May be used under the terms of the GNU Lesser General Public License (LGPL)
 * see http://www.gnu.org/copyleft/lgpl.html
 */
#pragma once
#include "../viterbi_decoder_core.h"
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <vector>
#include <immintrin.h>

// Vectorisation using SSE
//     8bit integers for errors, soft-decision values
//     16 way vectorisation from 128bits/8bits 
//     32bit decision type since 16 x 2 decisions bits per branch
template <size_t constraint_length, size_t code_rate>
class ViterbiDecoder_SSE_u8: public ViterbiDecoder_Core<constraint_length,code_rate,uint8_t,int8_t>
{
private:
    using Base = ViterbiDecoder_Core<constraint_length,code_rate,uint8_t,int8_t>;
    using decision_bits_t = typename Base::Decisions::format_t;
private:
    // Calculate the minimum constraint length for vectorisation
    // We require: stride(metric)/2 = stride(branch_table) = stride(decision)
    // total_states = 2^(K-1)
    //
    // sizeof(metric)       = total_states   * sizeof(u8)  = 2^(K-1)
    // sizeof(branch_table) = total_states/2 * sizeof(s8)  = 2^(K-2)
    // sizeof(decision)     = total_states   / 8           = 2^(K-4)
    //
    // sizeof(__m128i)      = 16 = 2^4
    // stride(metric)       = sizeof(metric)       / sizeof(__m128i) = 2^(K-5)
    // stride(branch_table) = sizeof(branch_table) / sizeof(__m128i) = 2^(K-6)
    // stride(decision)     = sizeof(decision)     / sizeof(u32)     = 2^(K-6)
    //
    // For stride(...) >= 1, then K >= 6
    static constexpr size_t SIMD_ALIGN = sizeof(__m128i);
    static constexpr size_t v_stride_metric = Base::Metrics::SIZE_IN_BYTES/SIMD_ALIGN;
    static constexpr size_t v_stride_branch_table = Base::BranchTable::SIZE_IN_BYTES/SIMD_ALIGN;
    static constexpr size_t v_stride_decision_bits = Base::Decisions::SIZE_IN_BYTES/SIMD_ALIGN; 
    static constexpr size_t K_min = 6;
    uint64_t m_renormalisation_bias;
public:
    static constexpr bool is_valid = Base::K >= K_min;

    template <typename ... U>
    ViterbiDecoder_SSE_u8(U&& ... args): Base(std::forward<U>(args)...) {
        static_assert(is_valid, "Insufficient constraint length for vectorisation");
        static_assert(Base::Metrics::ALIGNMENT % SIMD_ALIGN == 0);
        static_assert(Base::BranchTable::ALIGNMENT % SIMD_ALIGN == 0);
    }

    uint64_t get_error(const size_t end_state=0u) {
        auto* old_metric = Base::m_metrics.get_old();
        const uint16_t normalised_error = old_metric[end_state % Base::NUMSTATES];
        return m_renormalisation_bias + uint64_t(normalised_error);
    }

    void reset(const size_t starting_state = 0u) {
        Base::reset(starting_state);
        m_renormalisation_bias = uint64_t(0u);
    }

    void update(const int8_t* symbols, const size_t N) {
        // number of symbols must be a multiple of the code rate
        assert(N % Base::R == 0);
        const size_t total_decoded_bits = N / Base::R;
        const size_t max_decoded_bits = Base::get_traceback_length() + Base::TOTAL_STATE_BITS;
        assert((total_decoded_bits + Base::m_current_decoded_bit) <= max_decoded_bits);

        for (size_t s = 0; s < N; s+=Base::R) {
            auto* decision = Base::m_decisions[Base::m_current_decoded_bit];
            auto* old_metrics = Base::m_metrics.get_old();
            auto* new_metrics = Base::m_metrics.get_new();
            bfly(&symbols[s], decision, old_metrics, new_metrics);
            if (new_metrics[0] >= Base::m_config.renormalisation_threshold) {
                renormalise(new_metrics);
            }
            Base::m_metrics.swap();
            Base::m_current_decoded_bit++;
        }
    }
private:
    void bfly(const int8_t* symbols, decision_bits_t* decision, uint8_t* old_metrics, uint8_t* new_metrics) {
        const __m128i* v_branch_table = reinterpret_cast<const __m128i*>(Base::m_branch_table.data());
        __m128i* v_old_metrics = reinterpret_cast<__m128i*>(old_metrics);
        __m128i* v_new_metrics = reinterpret_cast<__m128i*>(new_metrics);
        uint32_t* v_decision = reinterpret_cast<uint32_t*>(decision);

        assert(uintptr_t(v_branch_table) % SIMD_ALIGN == 0);
        assert(uintptr_t(v_old_metrics)  % SIMD_ALIGN == 0);
        assert(uintptr_t(v_new_metrics)  % SIMD_ALIGN == 0);

        __m128i v_symbols[Base::R];

        // Vectorise constants
        for (size_t i = 0; i < Base::R; i++) {
            v_symbols[i] = _mm_set1_epi8(symbols[i]);
        }
        const __m128i max_error = _mm_set1_epi8(Base::m_config.soft_decision_max_error);

        for (size_t curr_state = 0u; curr_state < v_stride_branch_table; curr_state++) {
            // Total errors across R symbols
            __m128i total_error = _mm_set1_epi8(0);
            for (size_t i = 0u; i < Base::R; i++) {
                __m128i error = _mm_subs_epi8(v_branch_table[i*v_stride_branch_table+curr_state], v_symbols[i]);
                error = _mm_abs_epi8(error);
                total_error = _mm_adds_epu8(total_error, error);
            }

            // Butterfly algorithm
            const size_t curr_state_0 = curr_state;
            const size_t curr_state_1 = curr_state + v_stride_metric/2;
            const size_t next_state_0 = (curr_state << 1) | 0;
            const size_t next_state_1 = (curr_state << 1) | 1;

            const __m128i inverse_error = _mm_subs_epu8(max_error, total_error);
            const __m128i next_error_0_0 = _mm_adds_epu8(v_old_metrics[curr_state_0],   total_error);
            const __m128i next_error_1_0 = _mm_adds_epu8(v_old_metrics[curr_state_1], inverse_error);
            const __m128i next_error_0_1 = _mm_adds_epu8(v_old_metrics[curr_state_0], inverse_error);
            const __m128i next_error_1_1 = _mm_adds_epu8(v_old_metrics[curr_state_1],   total_error);

            const __m128i min_next_error_0 = _mm_min_epu8(next_error_0_0, next_error_1_0);
            const __m128i min_next_error_1 = _mm_min_epu8(next_error_0_1, next_error_1_1);
            const __m128i decision_0 = _mm_cmpeq_epi8(min_next_error_0, next_error_1_0);
            const __m128i decision_1 = _mm_cmpeq_epi8(min_next_error_1, next_error_1_1);

            // Update metrics
            v_new_metrics[next_state_0] = _mm_unpacklo_epi8(min_next_error_0, min_next_error_1);
            v_new_metrics[next_state_1] = _mm_unpackhi_epi8(min_next_error_0, min_next_error_1);

            // Pack decision bits 
            const uint32_t decision_bits_lo = uint32_t(_mm_movemask_epi8(_mm_unpacklo_epi8(decision_0, decision_1)));
            const uint32_t decision_bits_hi = uint32_t(_mm_movemask_epi8(_mm_unpackhi_epi8(decision_0, decision_1)));
            v_decision[curr_state] = uint32_t(decision_bits_hi << 16u) | decision_bits_lo;
        }
    }

    void renormalise(uint8_t* metric) {
        assert(uintptr_t(metric) % SIMD_ALIGN == 0);
        __m128i* v_metric = reinterpret_cast<__m128i*>(metric);

        // Find minimum  
        __m128i adjustv = v_metric[0];
        for (size_t i = 1u; i < v_stride_metric; i++) {
            adjustv = _mm_min_epu8(adjustv, v_metric[i]);
        }

        // Shift half of the array onto the other half and get the minimum between them
        // Repeat this until we get the minimum value of all 8bit values
        // NOTE: srli performs shift on 128bit lanes
        adjustv = _mm_min_epu8(adjustv, _mm_srli_si128(adjustv, 8));
        adjustv = _mm_min_epu8(adjustv, _mm_srli_si128(adjustv, 4));
        adjustv = _mm_min_epu8(adjustv, _mm_srli_si128(adjustv, 2));
        adjustv = _mm_min_epu8(adjustv, _mm_srli_si128(adjustv, 1));

        // Normalise to minimum
        const uint8_t* reduce_buffer = reinterpret_cast<uint8_t*>(&adjustv);
        const uint8_t min = reduce_buffer[0];
        const __m128i vmin = _mm_set1_epi8(min);
        for (size_t i = 0u; i < v_stride_metric; i++) {
            v_metric[i] = _mm_subs_epu8(v_metric[i], vmin);
        }

        // Keep track of absolute error metrics
        m_renormalisation_bias += uint64_t(min);
    }
};

