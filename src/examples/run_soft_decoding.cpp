#include "viterbi/viterbi_decoder.h"
#include "viterbi/viterbi_branch_table.h"
#include "viterbi/viterbi_decoder_scalar.h"
#include "viterbi/viterbi_decoder_avx.h"
#include "viterbi/viterbi_decoder_sse.h"

#include "viterbi/convolutional_encoder.h"
#include "viterbi/convolutional_encoder_lookup.h"
#include "viterbi/convolutional_encoder_shift_register.h"

#include "codes.h"
#include "test_helpers.h"
#include "getopt/getopt.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <vector>
#include <random>
#include <cmath>
#include <optional>
#include <chrono>

void init_test(
    const Code& code, 
    const int noise_level, 
    const size_t total_input_bytes, 
    std::optional<DecodeType> override_decode_type
);

void run_test(
    ViterbiDecoder<int16_t>* vitdec, 
    ConvolutionalEncoder* enc, 
    const int noise_level,
    const size_t total_input_bytes,
    const int16_t SOFT_DECISION_HIGH, 
    const int16_t SOFT_DECISION_LOW
);

void usage() {
    fprintf(stderr, 
        "run_soft_decoding, Runs viterbi decoding using 16bit soft decision values\n\n"
        "    [-c <code id> (default: 0)]\n"
        "    [-d <decode type> (default: 0)]\n"
        "        0: Default decoding type\n"
        "        1: Override with SCALAR\n"
        "        2: Override with SIMD_SSE\n"
        "        3: Override with SIMD_AVX\n"
        "    [-n <noise level> (default: 0) ]\n"
        "    [-s <random seed> (default: 0) ]\n"
        "    [-S Randomises seed ]\n"
        "    [-L <total input bytes> (default: 1024) ]\n"
        "    [-l Lists all available codes ]\n"
        "    [-h Show usage ]\n"
    );
}

int main(int argc, char** argv) {
    const size_t N_max = common_codes.size();
    assert(N_max > 0u);

    int config_type = 0;
    int override_id = 0;
    int noise_level = 0;
    int random_seed = 0;
    int total_input_bytes = 1024;
    bool is_randomise_seed = false;

	int opt; 
    while ((opt = getopt_custom(argc, argv, "c:d:n:s:Slh")) != -1) {
        switch (opt) {
        case 'c':
            config_type = atoi(optarg);
            break;
        case 'd':
            override_id = atoi(optarg);
            break;
        case 'n':
            noise_level = atoi(optarg);
            break;
        case 's':
            random_seed = atoi(optarg);
            break;
        case 'S':
            is_randomise_seed = true;
            break;
        case 'L':
            total_input_bytes = atoi(optarg);
            break;
        case 'l':
            list_codes(common_codes.data(), common_codes.size());
            return 0;
        case 'h':
        default:
            usage();
            return 0;
        }
    }

    if ((config_type < 0) || (config_type >= (int)N_max)) {
        fprintf(
            stderr, 
            "Config must be between %d...%d\n"
            "Run '%s -l' for list of codes\n", 
            0, int(N_max)-1, argv[0]);
        return 1;
    }

    if ((override_id < 0) || (override_id > 3)) {
        fprintf(
            stderr,
            "Override decoder argument must be between 0...3\n"
            "Run '%s -h' for list of valid arguments\n",
            argv[0]
        );
        return 1;
    }

    if (noise_level < 0) {
        fprintf(
            stderr,
            "Noise level must be positive\n"
        );
        return 1;
    }

    if (total_input_bytes < 0) {
        fprintf(stderr, "Total input bytes must be positive\n");
        return 1;
    }

    if (is_randomise_seed) {
        const auto dt_now = std::chrono::system_clock::now().time_since_epoch();
        const auto us_now = std::chrono::duration_cast<std::chrono::microseconds>(dt_now).count();
        std::srand((unsigned int)us_now);
        random_seed = std::rand();
        printf("Using random_seed=%d\n", random_seed);
    }
    std::srand((unsigned int)random_seed);

    std::optional<DecodeType> override_decode_type;
    switch (override_id) {
    case 1: override_decode_type = DecodeType::SCALAR;   break;
    case 2: override_decode_type = DecodeType::SIMD_SSE; break;
    case 3: override_decode_type = DecodeType::SIMD_AVX; break;
    case 0:                  
    default: 
        override_decode_type = {};  
        break;
    }

    const auto dtype = DecodeType::SIMD_AVX;
    const auto& code = common_codes[config_type];

    init_test(code, noise_level, size_t(total_input_bytes), override_decode_type);
    
    return 0;
}

void init_test(
    const Code& code, 
    const int noise_level, 
    const size_t total_input_bytes, 
    std::optional<DecodeType> override_decode_type
) {
    const auto* name = code.name.c_str();
    const size_t K = code.K;
    const size_t R = code.R;
    const auto* G = code.G.data();
    constexpr size_t K_max_lookup = 10;
    constexpr int16_t SOFT_DECISION_HIGH = +127;
    constexpr int16_t SOFT_DECISION_LOW  = -127;

    printf("Using '%s': K=%zu, R=%zu\n", name, K, R);

    DecodeType decode_type = code.decode_type;
    if (override_decode_type) {
        const auto new_decode_type = override_decode_type.value();
        if (new_decode_type > decode_type) {
            printf(
                "WARN: Selected code can only operate up to '%s' but '%s' was requested\n",
                get_decode_type_name(decode_type).c_str(),
                get_decode_type_name(new_decode_type).c_str()
            );
        } else {
            decode_type = new_decode_type;
        }
    }

    ViterbiDecoder<int16_t>* vitdec = NULL;
    ConvolutionalEncoder* enc = NULL;

    // Decide appropriate convolutional encoder
    if (K >= K_max_lookup) {
        printf(
            "Using shift register encoder due to large K (%zu >= %zu)\n",
            K, K_max_lookup
        );
        enc = new ConvolutionalEncoder_ShiftRegister(K, R, G);
    } else {
        enc = new ConvolutionalEncoder_Lookup(K, R, G);
    }

    // Create branch table with correct alignment 
    size_t simd_alignment = sizeof(int16_t);
    switch (decode_type) {
    case DecodeType::SIMD_SSE: simd_alignment = 16u; break;
    case DecodeType::SIMD_AVX: simd_alignment = 32u; break;
    default:                                         break;
    }
    auto branch_table = ViterbiBranchTable<int16_t>(K, R, G, SOFT_DECISION_HIGH, SOFT_DECISION_LOW, simd_alignment);

    // Create viterbi decoder
    switch (decode_type) {
    case DecodeType::SCALAR:
        {
            printf("Using SCALAR decoder\n");
            vitdec = new ViterbiDecoder_Scalar<uint16_t,int16_t>(branch_table);
        }
        break;
    case DecodeType::SIMD_SSE:
        {
            printf("Using SIMD_SSE with alignment=%zu\n", simd_alignment);
            vitdec = new ViterbiDecoder_SSE(branch_table);
        }
        break;
    case DecodeType::SIMD_AVX:
        {
            printf("Using SIMD_AVX with alignment=%zu\n", simd_alignment);
            vitdec = new ViterbiDecoder_AVX(branch_table);
        }
        break;
    default:
        printf("Unknown decoder type\n");
        break;
    }

    run_test(vitdec, enc, noise_level, total_input_bytes, SOFT_DECISION_HIGH, SOFT_DECISION_LOW);

    delete vitdec;
    delete enc;
}

void run_test(
    ViterbiDecoder<int16_t>* vitdec, 
    ConvolutionalEncoder* enc, 
    const int noise_level,
    const size_t total_input_bytes,
    const int16_t SOFT_DECISION_HIGH, 
    const int16_t SOFT_DECISION_LOW
) {
    assert(vitdec->K == enc->K);
    assert(vitdec->R == enc->R);

    const size_t K = vitdec->K; 
    const size_t R = vitdec->R; 

    const size_t total_input_bits = total_input_bytes*8u;
    vitdec->set_traceback_length(total_input_bits);

    // Generate test data
    std::vector<uint8_t> tx_input_bytes;
    std::vector<int16_t> output_symbols; 
    std::vector<uint8_t> rx_input_bytes;
    tx_input_bytes.resize(total_input_bytes);
    rx_input_bytes.resize(total_input_bytes);

    generate_random_bytes(tx_input_bytes.data(), tx_input_bytes.size());
    encode_data(enc, tx_input_bytes, output_symbols, SOFT_DECISION_HIGH, SOFT_DECISION_LOW);
    if (noise_level > 0) {
        add_noise(output_symbols.data(), output_symbols.size(), int16_t(noise_level));
    }
    clamp_vector(output_symbols.data(), output_symbols.size(), SOFT_DECISION_LOW, SOFT_DECISION_HIGH);

    const size_t total_output_symbols = output_symbols.size();
    vitdec->reset();
    vitdec->update(output_symbols.data(), total_output_symbols);
    const uint64_t error = vitdec->chainback(rx_input_bytes.data(), total_input_bits, 0u);
    printf("error=%zu\n", error);

    // Show decoding results
    const size_t total_errors = get_total_bit_errors(tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes);
    const float bit_error_rate = (float)total_errors / (float)total_input_bits * 100.0f;
    printf("bit error rate=%.2f%%\n", bit_error_rate);
    printf("%zu/%zu incorrect bits\n", total_errors, total_input_bits);
}

