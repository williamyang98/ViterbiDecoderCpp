/* Generic Viterbi decoder,
 * Copyright Phil Karn, KA9Q,
 * Karn's original code can be found here: https://github.com/ka9q/libfec 
 * May be used under the terms of the GNU Lesser General Public License (LGPL)
 * see http://www.gnu.org/copyleft/lgpl.html
 */
#pragma once
#include "./viterbi_decoder_core.h"
#include <stdint.h>
#include <stddef.h>
#include <vector>
#include <assert.h>
 
template <
    size_t constraint_length, size_t code_rate, typename error_t, typename soft_t, 
    typename decision_bits_t = uint64_t, typename absolute_error_t = uint64_t
>
class ViterbiDecoder_Scalar: public ViterbiDecoder_Core<constraint_length,code_rate,error_t,soft_t,decision_bits_t>
{
private:
    using Base = ViterbiDecoder_Core<constraint_length,code_rate,error_t,soft_t,decision_bits_t>;    
    static constexpr size_t K_min = 2;
    absolute_error_t m_renormalisation_bias;
public:
    static constexpr bool is_valid = Base::K >= K_min;

    template <typename ... U>
    ViterbiDecoder_Scalar(U&& ... args): Base(std::forward<U>(args)...) {
        static_assert(is_valid, "Scalar decoder must have constraint length of at least 2");
    }

    /// @brief Gets the error of the trajectory for a given end state.
    absolute_error_t get_error(const size_t end_state=0u) {
        auto* old_metric = Base::m_metrics.get_old();
        const error_t normalised_error = old_metric[end_state % Base::NUMSTATES];
        return m_renormalisation_bias + absolute_error_t(normalised_error);
    }

    /// @brief Resets the core data structures and our accumulated error from the last decode.
    void reset(const size_t starting_state = 0u) {
        Base::reset(starting_state);
        m_renormalisation_bias = absolute_error_t(0u);
    }

    /// @brief Given the output symbols of a convolutional code, start determining the lowest error trajectories through the trellis.
    void update(const soft_t* symbols, const size_t N) {
        // NOTE: We expect the symbol values to be in the range set by the branch_table
        //       symbols[i] ∈ [soft_decision_low, soft_decision_high]
        //       Otherwise when we calculate inside bfly(...):
        //           m_total_error = soft_decision_max_error - total_error
        //       The resulting value could underflow with unsigned error types 
        // number of symbols must be a multiple of the code rate
        assert(N % Base::R == 0);
        const size_t total_decoded_bits = N / Base::R;
        const size_t max_decoded_bits = Base::get_traceback_length() + Base::TOTAL_STATE_BITS;
        assert((total_decoded_bits + Base::m_current_decoded_bit) <= max_decoded_bits);

        for (size_t i = 0u; i < N; i+=(Base::R)) {
            auto* decision = Base::m_decisions[Base::m_current_decoded_bit];
            auto* old_metric = Base::m_metrics.get_old();
            auto* new_metric = Base::m_metrics.get_new();
            bfly(&symbols[i], decision, old_metric, new_metric);
            if (new_metric[0] >= Base::m_config.renormalisation_threshold) {
                renormalise(new_metric);
            }
            Base::m_metrics.swap();
            Base::m_current_decoded_bit++;
        }
    }
private:
    /// @brief Process R symbols and output 1 decoded bit
    void bfly(const soft_t* symbols, decision_bits_t* decision, error_t* old_metric, error_t* new_metric) {
        for (size_t curr_state = 0u; curr_state < Base::BranchTable::NUMSTATES; curr_state++) {
            // Error associated with state given symbols
            error_t total_error = 0u;
            for (size_t i = 0; i < Base::R; i++) {
                const soft_t sym = symbols[i];
                const soft_t expected_sym = Base::m_branch_table[i][curr_state];
                const soft_t error = expected_sym - sym;
                const error_t abs_error = error_t(get_abs(error));
                total_error += abs_error;
            }
            assert(total_error <= Base::m_config.soft_decision_max_error);

            // We only store half the states in the branch table, but here we expand it out to explore the other unstored half
            // Both state 0 and state 1 when shifted give the same next state (for the same input bit)
            // Most significant bit for state 1 is shifted out when considering the next state
            // since it is shifted out of the (K-1) bits considered for the next state
            const size_t curr_state_0 = curr_state;                               // Goes up to K-2 bits
            const size_t curr_state_1 = curr_state + Base::Metrics::NUMSTATES/2;  // Adds (K-1)th bit
            const size_t next_state_0 = (curr_state << 1) | 0;
            const size_t next_state_1 = (curr_state << 1) | 1;

            // This is the heart of the butterfly optimisation
            // There are 4 transitions between states we need to consider
            // Two starting states (0|X) and (1|X) with (K-1) bits
            // Two next states (X|0) and (X|1) with (K-1) bits
            // In the branch table we store the value for y = P{(0|X|0)^code} with (K) bits
            // Where P{x} is a function that gets the parity of the bit field
            // Note that P{x^y} = P{x}^P{y}
            //
            // Therefore for the transition (r|X) to (X|s) the output is
            // y' = P{(r|X|s)^code} = P{(r|0|s)}^P{(0|X|0)^code} = r^s^y
            // Depending on the r and s bit, the output symbol can be inverted
            //
            // Now consider error sum for a symbol
            // We assume that the symbols take the value L0..L1
            // Let y be the received symbol value
            // y lies on the line segment from L0..L1, which is L long
            // Let g be expected symbol L0 or L1, and g' be the inverse
            // g'<----------L----------------->g
            // g'<----e'-->y <-------e-------->g
            // e' = L-e'
            // For a block of R symbols that are either all inverted or not inverted
            // E' = sum(e') = LR - sum(e') = LR - E
            const error_t inverted_error = Base::m_config.soft_decision_max_error - total_error;

            // r = leading bit of previous state
            // s = input bit
            // g = parity bit corresponding to output symbol
            //  next_error_r_s = r^s^g
            const error_t next_error_0_0 = old_metric[curr_state_0] + total_error;
            const error_t next_error_1_0 = old_metric[curr_state_1] + inverted_error;
            const error_t next_error_0_1 = old_metric[curr_state_0] + inverted_error;
            const error_t next_error_1_1 = old_metric[curr_state_1] + total_error;
            // TODO: When adding our error metrics we may cause an overflow to happen 
            //       if the renormalisation step was not performed in time
            //       Our intrinsics implementations use saturated arithmetic to prevent an overflow
            //       Perhaps it is possible to use something like GCC's builtin saturated add here?

            // Select the previous state r with a lower error for an input bit s
            const decision_bits_t decision_0 = next_error_0_0 > next_error_1_0;
            const decision_bits_t decision_1 = next_error_0_1 > next_error_1_1;

            // Update metrics
            new_metric[next_state_0] = decision_0 ? next_error_1_0 : next_error_0_0;
            new_metric[next_state_1] = decision_1 ? next_error_1_1 : next_error_0_1;

            // Store the leading bit for the previous state for the next states (X|0) and (X|1)
            const decision_bits_t bits = decision_0 | (decision_1 << 1);
            const size_t curr_pack_index = next_state_0 / Base::Decisions::TOTAL_BITS_PER_BLOCK;
            const size_t curr_pack_bit   = next_state_0 % Base::Decisions::TOTAL_BITS_PER_BLOCK;
            decision[curr_pack_index] |= (bits << curr_pack_bit);
        }
    }

    /// @brief Normalise error metrics so minimum value is the numeric lower bound of the error type 
    void renormalise(error_t* metric) {
        error_t min = metric[0];
        for (size_t curr_state = 1u; curr_state < Base::Metrics::NUMSTATES; curr_state++) {
            error_t x = metric[curr_state];
            if (x < min) {
                min = x;
            }
        }

        for (size_t curr_state = 0u; curr_state < Base::Metrics::NUMSTATES; curr_state++) {
            metric[curr_state] -= min;
        }

        m_renormalisation_bias += absolute_error_t(min);
    }

    template <typename T>
    inline static
    T get_abs(T x) {
        return (x > 0) ? x : -x;
    }
};