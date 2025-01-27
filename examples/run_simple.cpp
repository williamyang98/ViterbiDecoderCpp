#include <stdio.h>
#include <stddef.h>
#include <inttypes.h>
#include <vector>
#include <limits>

#include "viterbi/convolutional_encoder_lookup.h"
#include "helpers/test_helpers.h"

#include "viterbi/viterbi_decoder_core.h"
#include "viterbi/viterbi_decoder_scalar.h"
// Uncomment to select different decoder
// #include "viterbi/x86/viterbi_decoder_sse_u16.h"
// #include "viterbi/x86/viterbi_decoder_avx_u16.h"
// #include "viterbi/arm/viterbi_decoder_neon_u16.h"

// NOTE: Other example code uses alot of templating to select different viterbi decoders at runtime
//       This example serves to demonstrate a simple use of the library if you just need one decoder type
int main(int argc, char** argv) {
    constexpr size_t K = 7;
    constexpr size_t R = 4;
    const uint8_t G[R] = { 109, 79, 83, 109 };

    // We are encoding each symbols as a 16bit value between -127 and +127
    const int16_t soft_decision_high = +127;
    const int16_t soft_decision_low  = -127;
    ViterbiDecoder_Config<uint16_t> decoder_config;
    {
        const uint16_t max_error = uint16_t(soft_decision_high-soft_decision_low) * uint16_t(R);
        const uint16_t error_margin = max_error * uint16_t(5u);
        decoder_config.soft_decision_max_error = max_error;
        decoder_config.initial_start_error = std::numeric_limits<uint16_t>::min();
        decoder_config.initial_non_start_error = decoder_config.initial_start_error + error_margin;
        decoder_config.renormalisation_threshold = std::numeric_limits<uint16_t>::max() - error_margin;
    }


    // Generate our data
    const size_t total_input_bytes = 1024;
    const size_t total_input_bits = total_input_bytes*8u;
    const size_t noise_level = 0;
    auto enc = ConvolutionalEncoder_Lookup(K, R, G);
    std::vector<uint8_t> tx_input_bytes;
    std::vector<int16_t> output_symbols; 
    tx_input_bytes.resize(total_input_bytes);
    {
        const size_t total_tail_bits = K-1u;
        const size_t total_data_bits = total_input_bytes*8;
        const size_t total_bits = total_data_bits + total_tail_bits;
        const size_t total_symbols = total_bits * R;
        output_symbols.resize(total_symbols);
    }
    generate_random_bytes(tx_input_bytes.data(), tx_input_bytes.size());
    encode_data(
        &enc, 
        tx_input_bytes.data(), tx_input_bytes.size(), 
        output_symbols.data(), output_symbols.size(),
        soft_decision_high, soft_decision_low
    );
    add_noise(output_symbols.data(), output_symbols.size(), noise_level);
    clamp_vector(output_symbols.data(), output_symbols.size(), soft_decision_low, soft_decision_high);


    // Decode the data
    std::vector<uint8_t> rx_input_bytes;
    rx_input_bytes.resize(total_input_bytes);
    auto branch_table = ViterbiBranchTable<K,R,int16_t>(G, soft_decision_high, soft_decision_low);
    auto vitdec = ViterbiDecoder_Core<K,R,uint16_t,int16_t>(branch_table, decoder_config);

    // NOTE: Up to you to choose your desired decoder type
    // using Decoder = ViterbiDecoder_AVX_u16<K,R>;
    // using Decoder = ViterbiDecoder_SSE_u16<K,R>;
    // using Decoder = ViterbiDecoder_NEON_u16<K,R>;
    using Decoder = ViterbiDecoder_Scalar<K,R,uint16_t,int16_t>;

    vitdec.set_traceback_length(total_input_bits);
    vitdec.reset();
    const uint64_t accumulated_error = Decoder::template update<uint64_t>(vitdec, output_symbols.data(), output_symbols.size());
    const uint64_t error = accumulated_error + uint64_t(vitdec.get_error());
    vitdec.chainback(rx_input_bytes.data(), total_input_bits);
    printf("error_metric=%" PRIu64 "\n", error);


    // Show decoding results
    const size_t total_errors = get_total_bit_errors(tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes);
    const float bit_error_rate = (float)total_errors / (float)total_input_bits * 100.0f;
    printf("bit_error_rate=%.2f%%\n", bit_error_rate);
    printf("%zu/%zu incorrect bits\n", total_errors, total_input_bits);
 
    if (total_errors > 0) {
        printf("ERROR: Simple example had decoding errors\n");
        return 1;
    }
    return 0;
}