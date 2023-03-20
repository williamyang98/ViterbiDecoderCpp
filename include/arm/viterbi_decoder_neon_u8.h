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
//     8bit integers for errors, soft-decision values
//     16 way vectorisation from 128bits/16bits 
//     32bit decision type since 16 x 2 decisions bits per branch
template <size_t constraint_length, size_t code_rate>
class ViterbiDecoder_NEON_u8: public ViterbiDecoder_Core<constraint_length, code_rate, uint8_t, int8_t, uint32_t>
{
private:
    using Base = ViterbiDecoder_Core<constraint_length, code_rate, uint8_t, int8_t, uint32_t>;
private:
    // metric:       NUMSTATES   * sizeof(u8)                       = NUMSTATES
    // branch_table: NUMSTATES/2 * sizeof(s8)                       = NUMSTATES/2
    // decision:     NUMSTATES/DECISION_BITSIZE * DECISION_BYTESIZE = NUMSTATES/8
    // 
    // m256_metric_width:       NUMSTATES   / sizeof(uint16x8_t) = NUMSTATES/16
    // m256_branch_table_width: NUMSTATES/2 / sizeof(uint16x8_t) = NUMSTATES/32
    // u32_decision_width:      NUMSTATES/8 / sizeof(u32)        = NUMSTATES/32
    static constexpr size_t ALIGN_AMOUNT = sizeof(uint16x8_t);
    static constexpr size_t m128_width_metric = Base::NUMSTATES/ALIGN_AMOUNT;
    static constexpr size_t m128_width_branch_table = Base::NUMSTATES/(2u*ALIGN_AMOUNT);
    static constexpr size_t u32_width_decision = Base::NUMSTATES/(2u*ALIGN_AMOUNT);
    static constexpr size_t K_min = 6;
    uint64_t renormalisation_bias;
public:
    static constexpr bool is_valid = Base::K >= K_min;

    template <typename ... U>
    ViterbiDecoder_NEON_u8(U&& ... args)
    :   Base(std::forward<U>(args)...)
    {
        static_assert(is_valid, "Insufficient constraint length for vectorisation");
        static_assert(Base::METRIC_ALIGNMENT % ALIGN_AMOUNT == 0);
        static_assert(Base::branch_table.alignment % ALIGN_AMOUNT == 0);
    }

    inline
    uint64_t get_error(const size_t end_state=0u) {
        auto* old_metric = Base::get_old_metric();
        const uint8_t normalised_error = old_metric[end_state % Base::NUMSTATES];
        return renormalisation_bias + uint64_t(normalised_error);
    }

    inline
    void reset(const size_t starting_state = 0u) {
        Base::reset(starting_state);
        renormalisation_bias = uint64_t(0u);
    }

    inline
    void update(const int8_t* symbols, const size_t N) {
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
    void bfly(const int8_t* symbols, uint32_t* decision, uint8_t* old_metric, uint8_t* new_metric) 
    {
        const int8x16_t* m128_branch_table = reinterpret_cast<const int8x16_t*>(Base::branch_table.data());
        uint8x16_t* m128_old_metric = reinterpret_cast<uint8x16_t*>(old_metric);
        uint8x16_t* m128_new_metric = reinterpret_cast<uint8x16_t*>(new_metric);

        assert(((uintptr_t)m128_branch_table % ALIGN_AMOUNT) == 0);
        assert(((uintptr_t)m128_old_metric % ALIGN_AMOUNT) == 0);
        assert(((uintptr_t)m128_new_metric % ALIGN_AMOUNT) == 0);

        int8x16_t m128_symbols[Base::R];

        // Vectorise constants
        for (size_t i = 0; i < Base::R; i++) {
            m128_symbols[i] = vmovq_n_s8(symbols[i]);
        }
        const uint8x16_t max_error = vmovq_n_u8(Base::config.soft_decision_max_error);

        for (size_t curr_state = 0u; curr_state < m128_width_branch_table; curr_state++) {
            // Total errors across R symbols
            uint8x16_t total_error = vmovq_n_u8(0);
            for (size_t i = 0u; i < Base::R; i++) {
                int8x16_t error = vabdq_s8(
                    m128_branch_table[i*m128_width_branch_table+curr_state], 
                    m128_symbols[i]
                );
                total_error = vqaddq_u8(total_error, vreinterpretq_u8_s8(error));
            }

            // Butterfly algorithm
            const uint8x16_t m_total_error = vqsubq_u8(max_error, total_error);
            const uint8x16_t m0 = vqaddq_u8(m128_old_metric[curr_state                      ],   total_error);
            const uint8x16_t m1 = vqaddq_u8(m128_old_metric[curr_state + m128_width_metric/2], m_total_error);
            const uint8x16_t m2 = vqaddq_u8(m128_old_metric[curr_state                      ], m_total_error);
            const uint8x16_t m3 = vqaddq_u8(m128_old_metric[curr_state + m128_width_metric/2],   total_error);
            const uint8x16_t survivor0 = vminq_u8(m0, m1);
            const uint8x16_t survivor1 = vminq_u8(m2, m3);
            const uint8x16_t decision0 = vceqq_u8(survivor0, m1);
            const uint8x16_t decision1 = vceqq_u8(survivor1, m3);

            // Update metrics
            m128_new_metric[2*curr_state+0] = vzip1q_u8(survivor0, survivor1);
            m128_new_metric[2*curr_state+1] = vzip2q_u8(survivor0, survivor1);

            // Pack decision bits
            decision[curr_state] = pack_decision_bits(decision0, decision1);
        }
    }

    inline
    void renormalise(uint8_t* metric) {
        assert(((uintptr_t)metric % ALIGN_AMOUNT) == 0);
        uint8x16_t* m128_metric = reinterpret_cast<uint8x16_t*>(metric);

        // Find minimum 
        uint8x16_t adjustv = m128_metric[0];
        for (size_t i = 1u; i < m128_width_metric; i++) {
            adjustv = vminq_u8(adjustv, m128_metric[i]);
        }

        // Shift half of the array onto the other half and get the minimum between them
        // TODO: Use intrinsics
        const uint8_t* buf = reinterpret_cast<const uint8_t*>(&adjustv);
        uint8_t min = buf[0];
        for (size_t i = 1; i < 16; i++) {
            const uint8_t v = buf[i];
            min = (min > v) ? v : min;
        }

        // Normalise to minimum
        const uint8x16_t vmin = vmovq_n_u8(min);
        for (size_t i = 0u; i < m128_width_metric; i++) {
            m128_metric[i] = vqsubq_u8(m128_metric[i], vmin);
        }

        // Keep track of absolute error metrics
        renormalisation_bias += uint64_t(min);
    }

    inline 
    uint32_t pack_decision_bits(uint8x16_t decision0, uint8x16_t decision1) {
        constexpr uint8_t ALIGNED(16) _d0_mask[16] = {
            1<<0, 1<<2, 1<<4, 1<<6,
            1<<0, 1<<2, 1<<4, 1<<6,
            1<<0, 1<<2, 1<<4, 1<<6,
            1<<0, 1<<2, 1<<4, 1<<6,
        };

        constexpr uint8_t ALIGNED(16) _d1_mask[16] = {
            1<<1, 1<<3, 1<<5, 1<<7,
            1<<1, 1<<3, 1<<5, 1<<7,
            1<<1, 1<<3, 1<<5, 1<<7,
            1<<1, 1<<3, 1<<5, 1<<7,
        };

        constexpr int32_t ALIGNED(16) _shift_mask[4] = {
            0, 8, 16, 24
        };

        uint8x16_t d0_mask = vld1q_u8(_d0_mask);
        uint8x16_t d1_mask = vld1q_u8(_d1_mask);
        int32x4_t shift_mask = vld1q_s32(_shift_mask);

        uint8x16_t m0 = vorrq_u8(vandq_u8(decision0, d0_mask), vandq_u8(decision1, d1_mask));
        uint16x8_t m1 = vpaddlq_u8(m0);
        uint32x4_t m2 = vpaddlq_u16(m1);

        uint32x4_t m3 = vshlq_u32(m2, shift_mask);
        uint64x2_t m4 = vpaddlq_u32(m3);

        uint64_t v = vgetq_lane_u64(m4, 0) | vgetq_lane_u64(m4, 1);
        return uint32_t(v);
    }
};

#undef ALIGNED