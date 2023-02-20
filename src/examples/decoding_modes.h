#pragma once

#include <stdint.h>
#include <stddef.h>
#include "viterbi/viterbi_decoder_core.h"

template <typename soft_t, typename error_t>
struct Decoder_Config {
    soft_t soft_decision_high;
    soft_t soft_decision_low;
    ViterbiDecoder_Config<error_t> decoder_config;
};

Decoder_Config<int16_t, uint16_t> get_soft16_decoding_config(const size_t code_rate) {
    const int16_t soft_decision_high = +127;
    const int16_t soft_decision_low  = -127;
    const uint16_t max_error = uint16_t(soft_decision_high-soft_decision_low) * uint16_t(code_rate);
    const uint16_t error_margin = max_error * uint16_t(5u);

    ViterbiDecoder_Config<uint16_t> config;
    config.soft_decision_max_error = max_error;
    config.initial_start_error = std::numeric_limits<uint16_t>::min();
    config.initial_non_start_error = config.initial_start_error + error_margin;
    config.renormalisation_threshold = std::numeric_limits<uint16_t>::max() - error_margin;

    return { soft_decision_high, soft_decision_low, config };
}

Decoder_Config<int8_t, uint8_t> get_soft8_decoding_config(const size_t code_rate) { 
    const int8_t soft_decision_high = +3;
    const int8_t soft_decision_low  = -3;
    const uint8_t max_error = uint8_t(soft_decision_high-soft_decision_low) * uint8_t(code_rate);
    const uint8_t error_margin = max_error * uint8_t(2u);

    ViterbiDecoder_Config<uint8_t> config;
    config.soft_decision_max_error = max_error;
    config.initial_start_error = std::numeric_limits<uint8_t>::min();
    config.initial_non_start_error = config.initial_start_error + error_margin;
    config.renormalisation_threshold = std::numeric_limits<uint8_t>::max() - error_margin;

    return { soft_decision_high, soft_decision_low, config };
}

Decoder_Config<int8_t, uint8_t> get_hard8_decoding_config(const size_t code_rate) {
    const int8_t soft_decision_high = +1;
    const int8_t soft_decision_low  = -1;
    const uint8_t max_error = uint8_t(soft_decision_high-soft_decision_low) * uint8_t(code_rate);
    const uint8_t error_margin = max_error * uint8_t(3u);

    ViterbiDecoder_Config<uint8_t> config;
    config.soft_decision_max_error = max_error;
    config.initial_start_error = std::numeric_limits<uint8_t>::min();
    config.initial_non_start_error = config.initial_start_error + error_margin;
    config.renormalisation_threshold = std::numeric_limits<uint8_t>::max() - error_margin;

    return { soft_decision_high, soft_decision_low, config };
}