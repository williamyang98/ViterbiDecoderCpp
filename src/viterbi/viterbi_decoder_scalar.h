/* Generic Viterbi decoder,
 * Copyright Phil Karn, KA9Q,
 * Karn's original code can be found here: https://github.com/ka9q/libfec 
 * May be used under the terms of the GNU Lesser General Public License (LGPL)
 * see http://www.gnu.org/copyleft/lgpl.html
 */
#pragma once

#include "viterbi_decoder.h"
#include "viterbi_branch_table.h"

#include <stdint.h>
#include <stddef.h>
#include <vector>
#include <assert.h>
#include "utility/parity_table.h"
#include "utility/aligned_vector.h"
#include "utility/basic_ops.h"

template <
    typename error_t = uint16_t,                // expects unsigned integer type
    typename soft_t = int16_t,                  // expects either unsigned or signed integer types
    typename decision_bits_t = uint64_t,        // expects unsigned integer type
    typename absolute_error_t = uint64_t        // expects unsigned integer type that is >= in size to error_t
>
class ViterbiDecoder_Scalar: public ViterbiDecoder<soft_t, absolute_error_t>
{
protected:
    static constexpr size_t DECISIONTYPE_BITSIZE = sizeof(decision_bits_t) * 8u;
    const size_t NUMSTATES;
    const size_t TOTAL_STATE_BITS;
    const size_t DECISION_BITS_LENGTH;
    const size_t METRIC_LENGTH;

    const ViterbiBranchTable<soft_t>& branch_table;   // we can reuse an existing branch table
    AlignedVector<error_t> metrics;             // shape: (2 x METRIC_LENGTH)      
    size_t curr_metric_index;                   // 0/1 to swap old and new metrics
    std::vector<decision_bits_t> decisions;     // shape: (TRACEBACK_LENGTH x DECISION_BITS_LENGTH)
    size_t curr_decoded_bit;

    const error_t soft_decision_max_error;      // max total error for R output symbols against reference
    static constexpr error_t INITIAL_START_ERROR = std::numeric_limits<error_t>::min();
    const error_t INITIAL_NON_START_ERROR;      
    const error_t RENORMALISATION_THRESHOLD;    // threshold to normalise all errors to 0
    absolute_error_t renormalisation_bias;      // keep track of the absolute error when we renormalise error_t
public:
    ViterbiDecoder_Scalar(
        const ViterbiBranchTable<soft_t>& _branch_table)
    :   ViterbiDecoder(_branch_table.K, _branch_table.R),
        // size of various data structures
        NUMSTATES(std::size_t(1) << (K-1u)),
        TOTAL_STATE_BITS(K-1u),
        DECISION_BITS_LENGTH(max(NUMSTATES/DECISIONTYPE_BITSIZE, std::size_t(1u))),
        METRIC_LENGTH(NUMSTATES),
        // internal data structures
        // NOTE: We align the metrics generously to 32bytes for possible AVX2 alignment
        branch_table(_branch_table),
        metrics(2*METRIC_LENGTH, 32u),  
        decisions(),
        // soft decision boundaries
        soft_decision_max_error(error_t(branch_table.soft_decision_high-branch_table.soft_decision_low) * error_t(R)),
        INITIAL_NON_START_ERROR(INITIAL_START_ERROR+soft_decision_max_error+1u),
        // TODO: The renormalisation check only tests the first error metric
        //       This can result in the other error metrics saturating or worse wrap around if we are using modular arithmetic
        //       To avoid this we set the renormalisation threshold to be quite generous, but it is not guaranteed to be strict
        //       Can we make this check safer or more reliable?
        RENORMALISATION_THRESHOLD(std::numeric_limits<error_t>::max()-soft_decision_max_error*10u)
    {
        assert(K == branch_table.K);
        assert(R == branch_table.R);
        assert(K > 1u);       
        assert(R > 1u);
        reset();
    }

    // Traceback length doesn't include tail bits
    virtual void set_traceback_length(const size_t traceback_length) {
        const size_t new_length = traceback_length + TOTAL_STATE_BITS;
        decisions.resize(new_length * DECISION_BITS_LENGTH);
        if (curr_decoded_bit > new_length) {
            curr_decoded_bit = new_length;
        }
    }

    virtual size_t get_traceback_length() const {
        const size_t N = decisions.size();
        const size_t M = N / DECISION_BITS_LENGTH;
        return M - TOTAL_STATE_BITS;
    }

    virtual void reset(const size_t starting_state = 0u) {
        curr_metric_index = 0u;
        curr_decoded_bit = 0u;
        renormalisation_bias = 0u;
        auto* old_metric = get_old_metric();
        for (size_t i = 0; i < METRIC_LENGTH; i++) {
            old_metric[i] = INITIAL_NON_START_ERROR;
        }
        const size_t STATE_MASK = NUMSTATES-1u;
        old_metric[starting_state & STATE_MASK] = INITIAL_START_ERROR;
        std::memset(decisions.data(), 0, decisions.size()*sizeof(decision_bits_t));
    }

    virtual absolute_error_t chainback(uint8_t* bytes_out, const size_t total_bits, const size_t end_state = 0u) {
        const size_t TRACEBACK_LENGTH = get_traceback_length();
        const auto [ADDSHIFT, SUBSHIFT] = get_shift();
        assert(TRACEBACK_LENGTH >= total_bits);
        assert((curr_decoded_bit - TOTAL_STATE_BITS) == total_bits);

        size_t curr_state = end_state;
        curr_state = (curr_state % NUMSTATES) << ADDSHIFT;

        for (size_t i = 0u; i < total_bits; i++) {
            const size_t j = (total_bits-1)-i;
            const size_t curr_decoded_byte = j/8;
            const size_t curr_decision = j + TOTAL_STATE_BITS;

            const size_t curr_pack_index = (curr_state >> ADDSHIFT) / DECISIONTYPE_BITSIZE;
            const size_t curr_pack_bit   = (curr_state >> ADDSHIFT) % DECISIONTYPE_BITSIZE;

            auto* decision = get_decision(curr_decision);
            const size_t input_bit = (decision[curr_pack_index] >> curr_pack_bit) & 0b1;

            curr_state = (curr_state >> 1) | (input_bit << (K-2+ADDSHIFT));
            bytes_out[curr_decoded_byte] = (uint8_t)(curr_state >> SUBSHIFT);
        }

        auto* old_metric = get_old_metric();
        const error_t normalised_error = old_metric[end_state % NUMSTATES];
        return renormalisation_bias + absolute_error_t(normalised_error);
    }

    // NOTE: We expect the symbol values to be in the range set by the branch_table
    //       symbols[i] ∈ [soft_decision_low, soft_decision_high]
    //       Otherwise when we calculate inside bfly(...):
    //           m_total_error = soft_decision_max_error - total_error
    //       The resulting value could underflow with unsigned error types 
    virtual void update(const soft_t* symbols, const size_t N) {
        // number of symbols must be a multiple of the code rate
        assert(N % R == 0);
        const size_t total_decoded_bits = N / R;
        const size_t max_decoded_bits = get_traceback_length() + TOTAL_STATE_BITS;
        assert((total_decoded_bits + curr_decoded_bit) <= max_decoded_bits);

        for (size_t i = 0u; i < N; i+=R) {
            auto* decision = get_decision(curr_decoded_bit);
            auto* old_metric = get_old_metric();
            auto* new_metric = get_new_metric();
            bfly(&symbols[i], decision, old_metric, new_metric);
            if (new_metric[0] >= RENORMALISATION_THRESHOLD) {
                renormalise(new_metric);
            }
            swap_metrics();
            curr_decoded_bit++;
        }
    }
protected:
    // swap old and new metrics
    inline
    error_t* get_new_metric() { 
        return &metrics[curr_metric_index]; 
    }

    inline
    error_t* get_old_metric() { 
        return &metrics[METRIC_LENGTH-curr_metric_index]; 
    }

    inline
    void swap_metrics() { 
        curr_metric_index = METRIC_LENGTH-curr_metric_index; 
    }

    inline
    decision_bits_t* get_decision(const size_t i) {
        return &decisions[i*DECISION_BITS_LENGTH];
    }
private:
    // Process R symbols for 1 decoded bit
    inline 
    void bfly(const soft_t* symbols, decision_bits_t* decision, error_t* old_metric, error_t* new_metric) {
        for (size_t curr_state = 0u; curr_state < branch_table.stride; curr_state++) {
            // Error associated with state given symbols
            error_t total_error = 0u;
            for (size_t i = 0; i < R; i++) {
                const soft_t sym = symbols[i];
                const soft_t expected_sym = branch_table[i][curr_state];
                const soft_t error = expected_sym - sym;
                const error_t abs_error = error_t(abs(error));
                total_error += abs_error;
            }
            assert(total_error <= soft_decision_max_error);

            // Butterfly algorithm
            const error_t m_total_error = soft_decision_max_error - total_error;
            const error_t m0 = old_metric[curr_state                  ] +   total_error;
            const error_t m1 = old_metric[curr_state + METRIC_LENGTH/2] + m_total_error;
            const error_t m2 = old_metric[curr_state                  ] + m_total_error;
            const error_t m3 = old_metric[curr_state + METRIC_LENGTH/2] +   total_error;
            const decision_bits_t d0 = m0 > m1;
            const decision_bits_t d1 = m2 > m3;

            // Update metrics
            new_metric[2*curr_state+0] = d0 ? m1 : m0;
            new_metric[2*curr_state+1] = d1 ? m3 : m2;

            // Pack decision bits
            const decision_bits_t bits = d0 | (d1 << 1);
            constexpr size_t total_decision_bits = 2u;
            const size_t curr_bit_index = curr_state*total_decision_bits;
            const size_t curr_pack_index = curr_bit_index / DECISIONTYPE_BITSIZE;
            const size_t curr_pack_bit   = curr_bit_index % DECISIONTYPE_BITSIZE;
            decision[curr_pack_index] |= (bits << curr_pack_bit);
        }
    }

    // Normalise error metrics so minimum value is 0
    inline 
    void renormalise(error_t* metric) {
        error_t min = metric[0];
        for (size_t curr_state = 1u; curr_state < METRIC_LENGTH; curr_state++) {
            error_t x = metric[curr_state];
            if (x < min) {
                min = x;
            }
        }

        for (size_t curr_state = 0u; curr_state < METRIC_LENGTH; curr_state++) {
            error_t& x = metric[curr_state];
            x -= min;
        }

        renormalisation_bias += absolute_error_t(min);
    }
    
    // align curr_state so we get output bytes
    std::pair<size_t, size_t> get_shift() {
        const size_t M = K-1;
        if (M < 8) {
            return { 8-M, 0 };
        } else if (M > 8) {
            return { 0, M-8 };
        } else {
            return { 0, 0 };
        }
    };
};