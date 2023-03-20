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
#include "arm_neon.h"

#ifdef _MSC_VER
#define ALIGNED(x) __declspec(align(x))
#else
#define ALIGNED(x) __attribute__ ((aligned(x)))
#endif

// Vectorisation using 128bit ARM
//     16bit integers for errors, soft-decision values
//     8 way vectorisation from 128bits/16bits 
//     16bit decision type since 8 x 2 decisions bits per branch
template <size_t constraint_length, size_t code_rate>
class ViterbiDecoder_NEON_u16: public ViterbiDecoder_Core<constraint_length, code_rate, uint16_t, int16_t, uint16_t>
{
private:
    using Base = ViterbiDecoder_Core<constraint_length, code_rate, uint16_t, int16_t, uint16_t>;
private:
    // metric:       NUMSTATES   * sizeof(u16)                      = NUMSTATES*2
    // branch_table: NUMSTATES/2 * sizeof(s16)                      = NUMSTATES  
    // decision:     NUMSTATES/DECISION_BITSIZE * DECISION_BYTESIZE = NUMSTATES/8
    // 
    // m256_metric_width:       NUMSTATES*2 / sizeof(uint16x8_t) = NUMSTATES/8
    // m256_branch_table_width: NUMSTATES   / sizeof(uint16x8_t) = NUMSTATES/16
    // u32_decision_width:      NUMSTATES/8 / sizeof(u16)     = NUMSTATES/16
    static constexpr size_t ALIGN_AMOUNT = sizeof(uint16x8_t);
    static constexpr size_t m128_width_metric = Base::NUMSTATES/ALIGN_AMOUNT*2u;
    static constexpr size_t m128_width_branch_table = Base::NUMSTATES/ALIGN_AMOUNT;
    static constexpr size_t u16_width_decision = Base::NUMSTATES/ALIGN_AMOUNT; 
    static constexpr size_t K_min = 5;
    uint64_t renormalisation_bias;
public:
    static constexpr bool is_valid = Base::K >= K_min;

    template <typename ... U>
    ViterbiDecoder_NEON_u16(U&& ... args)
    :   Base(std::forward<U>(args)...)
    {
        static_assert(is_valid, "Insufficient constraint length for vectorisation");
        static_assert(Base::METRIC_ALIGNMENT % ALIGN_AMOUNT == 0);
        static_assert(Base::branch_table.alignment % ALIGN_AMOUNT == 0);
    }

    inline
    uint64_t get_error(const size_t end_state=0u) {
        auto* old_metric = Base::get_old_metric();
        const uint16_t normalised_error = old_metric[end_state % Base::NUMSTATES];
        return renormalisation_bias + uint64_t(normalised_error);
    }

    inline
    void reset(const size_t starting_state = 0u) {
        Base::reset(starting_state);
        renormalisation_bias = uint64_t(0u);
    }

    inline
    void update(const int16_t* symbols, const size_t N) {
        // number of symbols must be a multiple of the code rate
        assert(N % Base::R == 0);
        const size_t total_decoded_bits = N / Base::R;
        const size_t max_decoded_bits = Base::get_traceback_length() + Base::TOTAL_STATE_BITS;
        assert((total_decoded_bits + Base::curr_decoded_bit) <= max_decoded_bits);

        for (size_t s = 0; s < N; s+=(Base::R)) {
            auto* decision = Base::get_decision(Base::curr_decoded_bit);
            auto* old_metric = Base::get_old_metric();
            auto* new_metric = Base::get_new_metric();
            bfly(&symbols[s], decision, old_metric, new_metric);
            if (new_metric[0] >= Base::config.renormalisation_threshold) {
                renormalise(new_metric);
            }
            Base::swap_metrics();
            Base::curr_decoded_bit++;
        }
    }
private:
    inline
    void bfly(const int16_t* symbols, uint16_t* decision, uint16_t* old_metric, uint16_t* new_metric) 
    {
        const int16x8_t* m128_branch_table = reinterpret_cast<const int16x8_t*>(Base::branch_table.data());
        uint16x8_t* m128_old_metric = reinterpret_cast<uint16x8_t*>(old_metric);
        uint16x8_t* m128_new_metric = reinterpret_cast<uint16x8_t*>(new_metric);

        assert(((uintptr_t)m128_branch_table % ALIGN_AMOUNT) == 0);
        assert(((uintptr_t)m128_old_metric % ALIGN_AMOUNT) == 0);
        assert(((uintptr_t)m128_new_metric % ALIGN_AMOUNT) == 0);

        int16x8_t m128_symbols[Base::R];

        // Vectorise constants
        for (size_t i = 0; i < Base::R; i++) {
            m128_symbols[i] = vmovq_n_s16(symbols[i]);
        }
        const uint16x8_t max_error = vmovq_n_u16(Base::config.soft_decision_max_error);

        for (size_t curr_state = 0u; curr_state < m128_width_branch_table; curr_state++) {
            // Total errors across R symbols
            uint16x8_t total_error = vmovq_n_u16(0);
            for (size_t i = 0u; i < Base::R; i++) {
                int16x8_t error = vabdq_s16(
                    m128_branch_table[i*m128_width_branch_table+curr_state], 
                    m128_symbols[i]
                );
                total_error = vqaddq_u16(total_error, vreinterpretq_u16_s16(error));
            }

            // Butterfly algorithm
            const uint16x8_t m_total_error = vqsubq_u16(max_error, total_error);
            const uint16x8_t m0 = vqaddq_u16(m128_old_metric[curr_state                      ],   total_error);
            const uint16x8_t m1 = vqaddq_u16(m128_old_metric[curr_state + m128_width_metric/2], m_total_error);
            const uint16x8_t m2 = vqaddq_u16(m128_old_metric[curr_state                      ], m_total_error);
            const uint16x8_t m3 = vqaddq_u16(m128_old_metric[curr_state + m128_width_metric/2],   total_error);
            const uint16x8_t survivor0 = vminq_u16(m0, m1);
            const uint16x8_t survivor1 = vminq_u16(m2, m3);
            const uint16x8_t decision0 = vceqq_u16(survivor0, m1);
            const uint16x8_t decision1 = vceqq_u16(survivor1, m3);

            // Update metrics
            m128_new_metric[2*curr_state+0] = vzip1q_u16(survivor0, survivor1);
            m128_new_metric[2*curr_state+1] = vzip2q_u16(survivor0, survivor1);

            // Pack decision bits
            decision[curr_state] = pack_decision_bits(decision0, decision1);
        }
    }

    inline
    void renormalise(uint16_t* metric) {
        assert(((uintptr_t)metric % ALIGN_AMOUNT) == 0);
        uint16x8_t* m128_metric = reinterpret_cast<uint16x8_t*>(metric);

        // Find minimum 
        uint16x8_t adjustv = m128_metric[0];
        for (size_t i = 1u; i < m128_width_metric; i++) {
            adjustv = vminq_u16(adjustv, m128_metric[i]);
        }

        // Shift half of the array onto the other half and get the minimum between them
        // TODO: Use intrinsics
        const uint16_t* buf = reinterpret_cast<const uint16_t*>(&adjustv);
        uint16_t min = buf[0];
        for (size_t i = 1; i < 8; i++) {
            const uint16_t v = buf[i];
            min = (min > v) ? v : min;
        }

        // Normalise to minimum
        const uint16x8_t vmin = vmovq_n_u16(min);
        for (size_t i = 0u; i < m128_width_metric; i++) {
            m128_metric[i] = vqsubq_u16(m128_metric[i], vmin);
        }

        // Keep track of absolute error metrics
        renormalisation_bias += uint64_t(min);
    }

    inline 
    uint16_t pack_decision_bits(uint16x8_t decision0, uint16x8_t decision1) {
        constexpr uint16_t ALIGNED(16) _d0_mask[8] = {
            1<<0, 1<<2, 1<<4, 1<<6, 1<<8, 1<<10, 1<<12, 1<<14
        };

        constexpr uint16_t ALIGNED(16) _d1_mask[8] = {
            1<<1, 1<<3, 1<<5, 1<<7, 1<<9, 1<<11, 1<<13, 1<<15
        };

        uint16x8_t d0_mask = vld1q_u16(_d0_mask);
        uint16x8_t d1_mask = vld1q_u16(_d1_mask);

        uint16x8_t m0 = vorrq_u16(vandq_u16(decision0, d0_mask), vandq_u16(decision1, d1_mask));
        uint32x4_t m1 = vpaddlq_u16(m0);
        uint64x2_t m2 = vpaddlq_u32(m1);

        uint64_t v = vgetq_lane_u64(m2, 0) | vgetq_lane_u64(m2, 1);
        return uint16_t(v);
    }
};

#undef ALIGNED