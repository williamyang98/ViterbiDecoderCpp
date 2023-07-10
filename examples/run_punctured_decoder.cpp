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
#include "viterbi/viterbi_decoder_core.h"

#include "helpers/decode_type.h"
#include "helpers/simd_type.h"
#include "helpers/puncture_code_helpers.h"
#include "helpers/test_helpers.h"
#include "utility/span.h"
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

template <class factory_t, typename soft_t, typename error_t>
void run_test(
    const Decoder_Config<soft_t,error_t>& config,
    const uint64_t noise_level, const bool is_soft_noise
);

template <typename soft_t>
size_t run_punctured_encoder(
    ConvolutionalEncoder* enc, 
    soft_t* output_symbols, const size_t max_output_symbols,
    const uint8_t* input_bytes, const size_t total_input_bytes
);

template <class decoder_t, typename soft_t, typename error_t>
uint64_t run_punctured_decoder(
    ViterbiDecoder_Core<K,R,error_t,soft_t>& vitdec, 
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

uint64_t get_normalised_noise_level(DecodeType decode_type, float noise_level, float soft_decision_high) {
    switch (decode_type) {
        case DecodeType::SOFT16: {
            const float soft_noise_multiplier = 5.5f;
            const uint64_t soft_noise_level = uint64_t(noise_level * soft_decision_high * soft_noise_multiplier);
            return soft_noise_level;
        }
        case DecodeType::SOFT8: {
            const float soft_noise_multiplier = 5.8f;
            const uint64_t soft_noise_level = uint64_t(noise_level * soft_decision_high * soft_noise_multiplier);
            return soft_noise_level;
        }
        case DecodeType::HARD8: {
            const uint64_t hard_noise_level = uint64_t(noise_level * 100.0f);
            return hard_noise_level;
        }
        default:
            return 0;
    }
}

bool get_is_soft_noise(DecodeType decode_type) {
    switch (decode_type) {
    case DecodeType::SOFT16: return true;
    case DecodeType::SOFT8:  return true;
    case DecodeType::HARD8:  return false;
    default:                 return true;
    }
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
    for (const auto& decode_type: Decode_Type_List) {
        SELECT_DECODE_TYPE(decode_type, {
            auto get_config = it0;
            using factory_t = it1;

            auto config = get_config(R);
            const uint64_t norm_noise = get_normalised_noise_level(decode_type, normalised_noise_level, float(config.soft_decision_high));
            const bool is_soft_noise = get_is_soft_noise(decode_type);
            printf(">>> Running %s decode type\n", get_decode_type_str(decode_type));
            run_test<factory_t>(config, norm_noise, is_soft_noise);
        });
    };
    return 0;
}

template <class factory_t, typename soft_t, typename error_t>
void run_test(
    const Decoder_Config<soft_t,error_t>& config,
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
        config.soft_decision_low, config.soft_decision_high,
        output_symbols.data(), output_symbols.size(),
        tx_input_bytes.data(), tx_input_bytes.size()
    );

    // Add noise
    if (noise_level > 0) {
        if (is_soft_noise) {
            add_noise(output_symbols.data(), output_symbols.size(), noise_level);
            clamp_vector(output_symbols.data(), output_symbols.size(), config.soft_decision_low, config.soft_decision_high);
        } else {
            add_binary_noise(output_symbols.data(), output_symbols.size(), noise_level, uint64_t(NOISE_MAX));
        }
    }

    // decoding
    auto branch_table = ViterbiBranchTable<K,R,soft_t>(G, config.soft_decision_high, config.soft_decision_low);
    auto vitdec = ViterbiDecoder_Core<K,R,error_t,soft_t>(branch_table, config.decoder_config);
    const soft_t unpunctured_value = 0;

    vitdec.set_traceback_length(total_data_bits);
    for (const auto& simd_type: SIMD_Type_List) {
        SELECT_FACTORY_ITEM(factory_t, simd_type, K, R, {
            using decoder_t = it;
            if constexpr(decoder_t::is_valid) {
                const uint64_t accumulated_error = run_punctured_decoder<decoder_t>(vitdec, unpunctured_value, output_symbols.data(), total_output_symbols);
                vitdec.chainback(rx_input_bytes.data(), total_data_bits, 0u);
                const uint64_t traceback_error = accumulated_error + uint64_t(vitdec.get_error());
                const size_t total_errors = get_total_bit_errors(tx_input_bytes.data(), rx_input_bytes.data(), total_data_bytes);
                const float bit_error_rate = (float)total_errors / (float)total_data_bits * 100.0f;

                printf("> %s results\n", get_simd_type_string(simd_type));
                printf("traceback_error=%" PRIu64 "\n", traceback_error);
                printf("bit error rate=%.2f%%\n", bit_error_rate);
                printf("%zu/%zu incorrect bits\n", total_errors, total_data_bits);
                printf("\n");
            }
        });
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

template <class decoder_t, typename soft_t, typename error_t>
uint64_t run_punctured_decoder(
    ViterbiDecoder_Core<K,R,error_t,soft_t>& vitdec, 
    const soft_t soft_decision_unpunctured,
    soft_t* output_symbols, const size_t total_output_symbols
) {
    auto output_symbols_buf = span_t(output_symbols, total_output_symbols);

    PuncturedDecodeResult res;
    vitdec.reset();
    uint64_t accumulated_error = 0;

    res = decode_punctured_symbols<decoder_t>(
        vitdec, soft_decision_unpunctured,
        output_symbols_buf.data(), output_symbols_buf.size(), 
        PI_16, PI_total_bits, 
        PI_total_bits*R*PI_16_total_count);
    accumulated_error += res.accumulated_error;
    output_symbols_buf = output_symbols_buf.front(res.index_punctured_symbol);

    res = decode_punctured_symbols<decoder_t>(
        vitdec, soft_decision_unpunctured,
        output_symbols_buf.data(), output_symbols_buf.size(), 
        PI_15, PI_total_bits, 
        PI_total_bits*R*PI_15_total_count);
    accumulated_error += res.accumulated_error;
    output_symbols_buf = output_symbols_buf.front(res.index_punctured_symbol);

    res = decode_punctured_symbols<decoder_t>(
        vitdec, soft_decision_unpunctured,
        output_symbols_buf.data(), output_symbols_buf.size(), 
        PI_X, 24, 
        24);
    accumulated_error += res.accumulated_error;
    output_symbols_buf = output_symbols_buf.front(res.index_punctured_symbol);

    assert(output_symbols_buf.size() == 0u);
    return accumulated_error;
}