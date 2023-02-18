/* Generic Viterbi decoder,
 * Copyright Phil Karn, KA9Q,
 * Karn's original code can be found here: https://github.com/ka9q/libfec 
 * May be used under the terms of the GNU Lesser General Public License (LGPL)
 * see http://www.gnu.org/copyleft/lgpl.html
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

// Common interface 
template <
    typename soft_t,                // Type to represent encoded symbols for soft decision decoding
    typename error_t = uint64_t     // Error metric after decoding and traceback
>
class ViterbiDecoder
{
public:
    const size_t K;
    const size_t R;
public:
    ViterbiDecoder(const size_t constraint_length, const size_t code_rate)
    :   K(constraint_length),
        R(code_rate)
    {
        assert(K > 1u);       
        assert(R > 1u);
    }

    virtual ~ViterbiDecoder() = default;

    // Traceback length doesn't include tail bits
    virtual void set_traceback_length(const size_t traceback_length) = 0;
    virtual size_t get_traceback_length() const = 0;

    // Reset for each new sequence of symbols
    virtual void reset(const size_t starting_state = 0u) = 0;

    // Backtracks and places bits into a byte buffer
    virtual error_t chainback(uint8_t* bytes_out, const size_t total_bits, const size_t end_state = 0u) = 0;

    // Forward updates the error metrics for N encoded symbols
    virtual void update(const soft_t* symbols, const size_t N) = 0;
};