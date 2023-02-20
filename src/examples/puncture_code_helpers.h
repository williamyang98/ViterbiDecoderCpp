#pragma once

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <vector>
#include "viterbi/convolutional_encoder.h"

// Reads symbols and depunctured the number of requested symbols
template <typename decoder_t, typename soft_t, typename input_t = soft_t>
size_t decode_punctured_symbols(
    decoder_t& decoder, 
    const soft_t unpunctured_symbol_value,
    const input_t* punctured_symbols, const size_t total_symbols,
    const bool* puncture_code, const size_t puncture_code_length,
    const size_t requested_output_symbols)
{
    const size_t R = decoder.R;
    auto symbols = std::vector<soft_t>(R);

    size_t index_punctured_symbol = 0;
    size_t index_puncture_code = 0;
    size_t index_output_symbol = 0;
    while (index_output_symbol < requested_output_symbols) {
        for (size_t i = 0u; i < R; i++) {
            soft_t& v = symbols[i];
            const bool is_punctured = puncture_code[index_puncture_code];
            if (is_punctured) {
                // NOTE: If our puncture code is invalid or we request too many symbols
                //       we may expect a punctured symbol when there isn't one
                //       Ideally this is caught during development but as a failsafe we exit early
                assert(index_punctured_symbol < total_symbols);
                if (index_punctured_symbol >= total_symbols) { 
                    return index_punctured_symbol;
                }
                v = soft_t(punctured_symbols[index_punctured_symbol]);
                index_punctured_symbol++;
            } else {
                v = unpunctured_symbol_value;
            }
            index_puncture_code = (index_puncture_code+1) % puncture_code_length;
            index_output_symbol++;
        }
        decoder.update(symbols.data(), R);
    }

    return index_punctured_symbol;
}

template <typename T>
size_t encode_punctured_data(
    ConvolutionalEncoder* enc, 
    const uint8_t* input_bytes, const size_t total_input_bytes,
    T* output_symbols, const size_t max_output_symbols,
    const bool* puncture_code, const size_t total_puncture_codes,
    const T soft_decision_high,
    const T soft_decision_low) 
{
    const size_t K = enc->K;
    const size_t R = enc->R;

    const size_t total_input_bits = total_input_bytes*8;
    const size_t max_data_symbols = total_input_bits*R;

    size_t curr_output_symbol = 0u;
    size_t curr_puncture_code = 0u;
    auto push_symbols = [&](const uint8_t* buf, const size_t total_bits) {
        for (size_t i = 0u; i < total_bits; i++) {
            const size_t curr_byte = i / 8;
            const size_t curr_bit = i % 8;
            const bool bit = (buf[curr_byte] >> curr_bit) & 0b1;
            const bool is_punctured = puncture_code[curr_puncture_code];
            curr_puncture_code = (curr_puncture_code+1) % total_puncture_codes;

            if (is_punctured) {
                assert(curr_output_symbol < max_output_symbols);
                output_symbols[curr_output_symbol] = bit ? soft_decision_high : soft_decision_low;
                curr_output_symbol++;                
            }
        }
    };

    // encode input bytes
    auto symbols = std::vector<uint8_t>(R);
    for (size_t i = 0u; i < total_input_bytes; i++) {
        const uint8_t x = input_bytes[i];
        enc->consume_byte(x, symbols.data());
        push_symbols(symbols.data(), 8u*R);
    }

    return curr_output_symbol;
}

template <typename T>
size_t encode_punctured_tail(
    ConvolutionalEncoder* enc, 
    T* output_symbols, const size_t max_output_symbols,
    const bool* puncture_code, const size_t total_puncture_codes,
    const T soft_decision_high,
    const T soft_decision_low) 
{
    const size_t K = enc->K;
    const size_t R = enc->R;

    const size_t total_tail_bits = K-1;
    const size_t max_tail_symbols = total_tail_bits*R;

    size_t curr_output_symbol = 0u;
    size_t curr_puncture_code = 0u;
    auto push_symbols = [&](const uint8_t* buf, const size_t total_bits) {
        for (size_t i = 0u; i < total_bits; i++) {
            const size_t curr_byte = i / 8;
            const size_t curr_bit = i % 8;
            const bool bit = (buf[curr_byte] >> curr_bit) & 0b1;
            const bool is_punctured = puncture_code[curr_puncture_code];
            curr_puncture_code = (curr_puncture_code+1) % total_puncture_codes;

            if (is_punctured) {
                assert(curr_output_symbol < max_output_symbols);
                output_symbols[curr_output_symbol] = bit ? soft_decision_high : soft_decision_low;
                curr_output_symbol++;                
            }
        }
    };

    // terminate tail at state 0
    auto symbols = std::vector<uint8_t>(R);
    for (size_t i = 0u; i < total_tail_bits; ) {
        const size_t remain_bits = total_tail_bits-i;
        const size_t total_bits = min(remain_bits, size_t(8u));
        enc->consume_byte(0x00, symbols.data());
        push_symbols(symbols.data(), total_bits*R);
        i += total_bits;
    }

    return curr_output_symbol;
}