#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <vector>
#include <random>
#include <chrono>

#include "viterbi/convolutional_encoder_lookup.h"

#include "decoder_factories.h"
#include "decoding_modes.h"
#include "puncture_code_helpers.h"
#include "test_helpers.h"
#include "span.h"
#include "getopt/getopt.h"

// DAB radio convolutional code
// DOC: ETSI EN 300 401
// Clause 11.1 - Convolutional code
// Clause 11.1.1 - Mother code
// Octal form | Binary form | Reversed binary | Decimal form |
//     133    | 001 011 011 |    110 110 1    |      109     |
//     171    | 001 111 001 |    100 111 1    |       79     |
//     145    | 001 100 101 |    101 001 1    |       83     |
//     133    | 001 011 011 |    110 110 1    |      109     |
constexpr size_t K = 7; 
constexpr size_t R = 4;
const uint8_t G[R] = { 109, 79, 83, 109 };

// DOC: ETSI EN 300 401
// Clause 11.1.2 - Puncturing procedure
// Table 13 - Puncturing vectors for the PI_TABLE
// We will be using real punctures codes used in the DAB radio standard
const bool PI_TABLE[24][32] = {
    {1,1,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0},
    {1,1,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0},
    {1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0},
    {1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0},
    {1,1,0,0, 1,1,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,0,0,0},
    {1,1,0,0, 1,1,0,0, 1,1,0,0, 1,0,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,0,0,0},
    {1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,0,0,0},
    {1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0},
    {1,1,1,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0},
    {1,1,1,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,1,0, 1,1,0,0, 1,1,0,0, 1,1,0,0},
    {1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0, 1,1,0,0, 1,1,0,0},
    {1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0},
    {1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,0,0},
    {1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,0,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,0,0},
    {1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,0,0},
    {1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0},
    {1,1,1,1, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0},
    {1,1,1,1, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,1, 1,1,1,0, 1,1,1,0, 1,1,1,0},
    {1,1,1,1, 1,1,1,0, 1,1,1,1, 1,1,1,0, 1,1,1,1, 1,1,1,0, 1,1,1,0, 1,1,1,0},
    {1,1,1,1, 1,1,1,0, 1,1,1,1, 1,1,1,0, 1,1,1,1, 1,1,1,0, 1,1,1,1, 1,1,1,0},
    {1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,0, 1,1,1,1, 1,1,1,0, 1,1,1,1, 1,1,1,0},
    {1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,0, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,0},
    {1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,0},
    {1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1},
};
// 24bit tailbiting puncture
const bool PI_X [24] = {1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0, 1,1,0,0};

// DAB fast information channel puncture codes
// DOC: ETSI EN 300 401
// Clause 11.2 - Coding in the fast information channel
// PI_16, PI_15 and PI_X are used
const bool* PI_16 = PI_TABLE[16-1u];    // 32bit puncture
const bool* PI_15 = PI_TABLE[15-1u];    // 32bit puncture
constexpr size_t PI_total_bits = 32;       
constexpr size_t PI_16_total_count = 21;
constexpr size_t PI_15_total_count = 3;

constexpr int NOISE_MAX = 100;

template <typename soft_t>
struct SoftDecisionParameters {
    soft_t high;
    soft_t low;
    soft_t unpunctured;
};

template <class decoder_factory_t, typename soft_t, typename error_t>
void run_test(
    const ViterbiDecoder_Config<error_t>& config,
    const SoftDecisionParameters<soft_t>& soft_decision,
    const uint64_t noise_level, const bool is_soft_noise
);

template <typename soft_t>
size_t run_punctured_encoder(
    ConvolutionalEncoder* enc, 
    soft_t* output_symbols, const size_t max_output_symbols,
    const uint8_t* input_bytes, const size_t total_input_bytes
);

template <typename soft_t, class decoder_t>
void run_punctured_decoder(
    decoder_t& vitdec, 
    const soft_t soft_decision_unpunctured,
    soft_t* output_symbols, const size_t total_output_symbols
);

void usage() {
    fprintf(stderr, 
        "run_punctured_decoder, Runs viterbi decoder with puncturing on DAB radio code\n\n"
        "    [-n <normalised noise level> (default: 0)]\n"
        "        A value between 0 and 100\n"
        "        0   = No noise"
        "        100 = Maximum noise"
        "    [-s <random seed> (default: Random)]\n"
        "    [-h Show usage]\n"
    );
}

int main(int argc, char** argv) {
    int noise_level = 0;
    int random_seed = 0;
    bool is_randomise_seed = true;

	int opt; 
    while ((opt = getopt_custom(argc, argv, "n:s:h")) != -1) {
        switch (opt) {
        case 'n':
            noise_level = atoi(optarg);
            break;
        case 's':
            is_randomise_seed = false;
            random_seed = atoi(optarg);
            break;
        case 'h':
        default:
            usage();
            return 0;
        }
    }

    // Other arguments
    if ((noise_level < 0) || (noise_level > 100)) {
        fprintf(
            stderr,
            "Noise level must be between 0...100\n"
        );
        return 1;
    }


    // Generate seed
    if (is_randomise_seed) {
        const auto dt_now = std::chrono::system_clock::now().time_since_epoch();
        const auto us_now = std::chrono::duration_cast<std::chrono::microseconds>(dt_now).count();
        std::srand((unsigned int)us_now);
        random_seed = std::rand();
        printf("Using random_seed=%d\n", random_seed);
    }
    std::srand((unsigned int)random_seed);

    const float normalised_noise_level = float(noise_level) / 100.0f;
    {
        auto config = get_soft16_decoding_config(R);
        SoftDecisionParameters<int16_t> soft_decision;
        soft_decision.low = config.soft_decision_low;
        soft_decision.high = config.soft_decision_high;
        soft_decision.unpunctured = 0; 
        const float soft_noise_multiplier = 5.5f;
        const uint64_t soft_noise_level = uint64_t(normalised_noise_level * float(soft_decision.high) * soft_noise_multiplier);
        printf(">> Using soft_16 decoders\n");
        run_test<ViterbiDecoder_Factory_u16<K,R>>(
            config.decoder_config, 
            soft_decision, 
            soft_noise_level, true
        );
    }
    printf("\n");
    {
        auto config = get_soft8_decoding_config(R);
        SoftDecisionParameters<int8_t> soft_decision;
        soft_decision.low = config.soft_decision_low;
        soft_decision.high = config.soft_decision_high;
        soft_decision.unpunctured = 0; 
        const float soft_noise_multiplier = 5.8f;
        const uint64_t soft_noise_level = uint64_t(normalised_noise_level * float(soft_decision.high) * soft_noise_multiplier);
        printf(">> Using soft_8 decoders\n");
        run_test<ViterbiDecoder_Factory_u8<K,R>>(
            config.decoder_config, 
            soft_decision, 
            soft_noise_level, true
        );
    }
    printf("\n");
    {
        auto config = get_hard8_decoding_config(R);
        SoftDecisionParameters<int8_t> soft_decision;
        soft_decision.low = config.soft_decision_low;
        soft_decision.high = config.soft_decision_high;
        soft_decision.unpunctured = 0; 
        const uint64_t hard_noise_level = uint64_t(normalised_noise_level * 100.0f);
        printf(">> Using hard_8 decoders\n");
        run_test<ViterbiDecoder_Factory_u8<K,R>>(
            config.decoder_config, 
            soft_decision, 
            hard_noise_level, false
        );
    } 

    return 0;
}

template <class decoder_factory_t, typename soft_t, typename error_t>
void run_test(
    const ViterbiDecoder_Config<error_t>& config,
    const SoftDecisionParameters<soft_t>& soft_decision,
    const uint64_t noise_level, const bool is_soft_noise
) {
    // Generate data
    const size_t total_data_bits = PI_total_bits*PI_16_total_count 
                                 + PI_total_bits*PI_15_total_count;
    const size_t total_data_bytes = total_data_bits/8u;
    const size_t total_tail_bits = K-1;
    const size_t total_bits = total_data_bits + total_tail_bits;
    const size_t max_output_symbols = total_bits*R;

    auto tx_input_bytes = std::vector<uint8_t>(total_data_bytes);
    auto rx_input_bytes = std::vector<uint8_t>(total_data_bytes);
    auto output_symbols = std::vector<soft_t>(max_output_symbols);

    generate_random_bytes(tx_input_bytes.data(), tx_input_bytes.size());
    auto enc = ConvolutionalEncoder_Lookup(K, R, G);

    // punctured encoding
    const size_t total_output_symbols = run_punctured_encoder(
        &enc, 
        soft_decision.low, soft_decision.high,
        output_symbols.data(), output_symbols.size(),
        tx_input_bytes.data(), tx_input_bytes.size()
    );

    // Add noise
    if (noise_level > 0) {
        if (is_soft_noise) {
            add_noise(output_symbols.data(), output_symbols.size(), noise_level);
            clamp_vector(output_symbols.data(), output_symbols.size(), soft_decision.low, soft_decision.high);
        } else {
            add_binary_noise(output_symbols.data(), output_symbols.size(), noise_level, uint64_t(NOISE_MAX));
        }
    }

    // decoding
    auto branch_table = ViterbiBranchTable<K,R,soft_t>(G, soft_decision.high, soft_decision.low);

    if constexpr(decoder_factory_t::Scalar::is_valid) {
        auto vitdec = typename decoder_factory_t::Scalar(branch_table, config);
        vitdec.set_traceback_length(total_data_bits);
        run_punctured_decoder(vitdec, soft_decision.unpunctured, output_symbols.data(), total_output_symbols);
        vitdec.chainback(rx_input_bytes.data(), total_data_bits, 0u);
        const uint64_t traceback_error = vitdec.get_error();
        const size_t total_errors = get_total_bit_errors(tx_input_bytes.data(), rx_input_bytes.data(), total_data_bytes);
        const float bit_error_rate = (float)total_errors / (float)total_data_bits * 100.0f;

        printf("> Scalar results\n");
        printf("traceback_error=%" PRIu64 "\n", traceback_error);
        printf("bit error rate=%.2f%%\n", bit_error_rate);
        printf("%zu/%zu incorrect bits\n", total_errors, total_data_bits);
        printf("\n");
    }
    
    if constexpr(decoder_factory_t::SIMD_SSE::is_valid) {
        auto vitdec = typename decoder_factory_t::SIMD_SSE(branch_table, config);
        vitdec.set_traceback_length(total_data_bits);
        run_punctured_decoder(vitdec, soft_decision.unpunctured, output_symbols.data(), total_output_symbols);
        vitdec.chainback(rx_input_bytes.data(), total_data_bits, 0u);
        const uint64_t traceback_error = vitdec.get_error();
        const size_t total_errors = get_total_bit_errors(tx_input_bytes.data(), rx_input_bytes.data(), total_data_bytes);
        const float bit_error_rate = (float)total_errors / (float)total_data_bits * 100.0f;

        printf("> SIMD_SSE results\n");
        printf("traceback_error=%" PRIu64 "\n", traceback_error);
        printf("bit error rate=%.2f%%\n", bit_error_rate);
        printf("%zu/%zu incorrect bits\n", total_errors, total_data_bits);
        printf("\n");
    }

    if constexpr(decoder_factory_t::SIMD_AVX::is_valid) {
        auto vitdec = typename decoder_factory_t::SIMD_AVX(branch_table, config);
        vitdec.set_traceback_length(total_data_bits);
        run_punctured_decoder(vitdec, soft_decision.unpunctured, output_symbols.data(), total_output_symbols);
        vitdec.chainback(rx_input_bytes.data(), total_data_bits, 0u);
        const uint64_t traceback_error = vitdec.get_error();
        const size_t total_errors = get_total_bit_errors(tx_input_bytes.data(), rx_input_bytes.data(), total_data_bytes);
        const float bit_error_rate = (float)total_errors / (float)total_data_bits * 100.0f;

        printf("> SIMD_AVX results\n");
        printf("traceback_error=%" PRIu64 "\n", traceback_error);
        printf("bit error rate=%.2f%%\n", bit_error_rate);
        printf("%zu/%zu incorrect bits\n", total_errors, total_data_bits);
        printf("\n");
    }
}

template <typename soft_t>
size_t run_punctured_encoder(
    ConvolutionalEncoder* enc, 
    const soft_t soft_decision_low, const soft_t soft_decision_high,
    soft_t* output_symbols, const size_t max_output_symbols,
    const uint8_t* input_bytes, const size_t total_input_bytes
) {
    size_t total_output_symbols = 0u;
    {
        auto output_symbols_buf = span_t<soft_t>(output_symbols, max_output_symbols);
        auto input_bytes_buf = span_t<const uint8_t>(input_bytes, total_input_bytes);
        for (size_t i = 0u; i < PI_16_total_count; i++) {
            const size_t total_bytes = PI_total_bits/8u;
            const size_t N = encode_punctured_data(
                enc,
                input_bytes_buf.data(), total_bytes,
                output_symbols_buf.data(), output_symbols_buf.size(),
                PI_16, PI_total_bits,
                soft_decision_high, soft_decision_low
            );
            output_symbols_buf = output_symbols_buf.front(N);
            input_bytes_buf = input_bytes_buf.front(total_bytes);
            total_output_symbols += N;
        }

        for (size_t i = 0u; i < PI_15_total_count; i++) {
            const size_t total_bytes = PI_total_bits/8u;
            const size_t N = encode_punctured_data(
                enc,
                input_bytes_buf.data(), total_bytes,
                output_symbols_buf.data(), output_symbols_buf.size(),
                PI_15, PI_total_bits,
                soft_decision_high, soft_decision_low
            );
            output_symbols_buf = output_symbols_buf.front(N);
            input_bytes_buf = input_bytes_buf.front(total_bytes);
            total_output_symbols += N;
        }

        {
            const size_t N = encode_punctured_tail(
                enc,
                output_symbols_buf.data(), output_symbols_buf.size(),
                PI_X, 24,
                soft_decision_high, soft_decision_low
            );
            total_output_symbols += N;
        }

        assert(input_bytes_buf.size() == 0);
    }

    return total_output_symbols;
}

template <typename soft_t, class decoder_t>
void run_punctured_decoder(
    decoder_t& vitdec, 
    const soft_t soft_decision_unpunctured,
    soft_t* output_symbols, const size_t total_output_symbols
) {
    auto output_symbols_buf = span_t(output_symbols, total_output_symbols);
    size_t total_read_symbols = 0;

    vitdec.reset();

    total_read_symbols = decode_punctured_symbols(
        vitdec, soft_decision_unpunctured,
        output_symbols_buf.data(), output_symbols_buf.size(), 
        PI_16, PI_total_bits, 
        PI_total_bits*R*PI_16_total_count);
    output_symbols_buf = output_symbols_buf.front(total_read_symbols);

    total_read_symbols = decode_punctured_symbols(
        vitdec, soft_decision_unpunctured,
        output_symbols_buf.data(), output_symbols_buf.size(), 
        PI_15, PI_total_bits, 
        PI_total_bits*R*PI_15_total_count);
    output_symbols_buf = output_symbols_buf.front(total_read_symbols);

    total_read_symbols = decode_punctured_symbols(
        vitdec, soft_decision_unpunctured,
        output_symbols_buf.data(), output_symbols_buf.size(), 
        PI_X, 24, 
        24);
    output_symbols_buf = output_symbols_buf.front(total_read_symbols);

    assert(output_symbols_buf.size() == 0u);
}