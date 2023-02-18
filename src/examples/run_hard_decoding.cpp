#include "viterbi/viterbi_decoder.h"
#include "viterbi/viterbi_branch_table.h"
#include "viterbi/viterbi_decoder_scalar.h"

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
#include <chrono>

constexpr int NOISE_MAX = 100;

void init_test(
    const Code& code, 
    const int noise_level, 
    const size_t total_input_bytes
);

void run_test(
    ViterbiDecoder<int8_t>* vitdec, 
    ConvolutionalEncoder* enc, 
    const int noise_level,
    const size_t total_input_bytes,
    const int8_t SOFT_DECISION_HIGH, 
    const int8_t SOFT_DECISION_LOW
);

void usage() {
    fprintf(stderr, 
        "run_hard_decoding, Runs viterbi decoding using hard decision values\n\n"
        "    [-c <code id> (default: 0)]\n"
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
    int noise_level = 0;
    int random_seed = 0;
    bool is_randomise_seed = false;
    int total_input_bytes = 1024;

	int opt; 
    while ((opt = getopt_custom(argc, argv, "c:n:s:SLlh")) != -1) {
        switch (opt) {
        case 'c':
            config_type = atoi(optarg);
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

    if ((noise_level < 0) || (noise_level > NOISE_MAX)) {
        fprintf(
            stderr,
            "Noise level must be between %d...%d\n",
            0, NOISE_MAX
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

    const auto& code = common_codes[config_type];
    init_test(code, noise_level, size_t(total_input_bytes));
    
    return 0;
}

void init_test(
    const Code& code, 
    const int noise_level, 
    const size_t total_input_bytes
) {
    const auto* name = code.name.c_str();
    const size_t K = code.K;
    const size_t R = code.R;
    const auto* G = code.G.data();
    constexpr int8_t HARD_DECISION_HIGH = +1;
    constexpr int8_t HARD_DECISION_LOW  = -1;
    constexpr size_t K_max_lookup = 10;

    printf("Using '%s': K=%zu, R=%zu\n", name, K, R);
    printf("Using SCALAR decoder for hard decision decoding\n");

    // Create suitable convolutional encoder
    ConvolutionalEncoder* enc = NULL;
    if (K >= K_max_lookup) {
        printf(
            "Using shift register encoder due to large K (%zu >= %zu)\n",
            K, K_max_lookup
        );
        enc = new ConvolutionalEncoder_ShiftRegister(K, R, G);
    } else {
        enc = new ConvolutionalEncoder_Lookup(K, R, G);
    }

    // Create scalar viterbi decoder and branch table
    auto branch_table = ViterbiBranchTable<int8_t>(K, R, G, HARD_DECISION_HIGH, HARD_DECISION_LOW);
    auto vitdec = ViterbiDecoder_Scalar<uint8_t,int8_t>(branch_table);
    run_test(&vitdec, enc, noise_level, total_input_bytes, HARD_DECISION_HIGH, HARD_DECISION_LOW);
    delete enc;
}

void run_test(
    ViterbiDecoder<int8_t>* vitdec, 
    ConvolutionalEncoder* enc, 
    const int noise_level,
    const size_t total_input_bytes,
    const int8_t SOFT_DECISION_HIGH, 
    const int8_t SOFT_DECISION_LOW
) {
    assert(vitdec->K == enc->K);
    assert(vitdec->R == enc->R);

    const size_t K = vitdec->K; 
    const size_t R = vitdec->R; 

    const size_t total_input_bits = total_input_bytes*8u;
    vitdec->set_traceback_length(total_input_bits);

    // Generate test data
    std::vector<uint8_t> tx_input_bytes;
    std::vector<int8_t> output_symbols; 
    std::vector<uint8_t> rx_input_bytes;
    tx_input_bytes.resize(total_input_bytes);
    rx_input_bytes.resize(total_input_bytes);

    generate_random_bytes(tx_input_bytes.data(), tx_input_bytes.size());
    encode_data(enc, tx_input_bytes, output_symbols, SOFT_DECISION_HIGH, SOFT_DECISION_LOW);

    // flip bits at random
    if (noise_level > 0) {
        for (auto& v: output_symbols) {
            const int N = (std::rand() % NOISE_MAX) + NOISE_MAX;
            if (N < noise_level)  {
                v = -v;
            }
        }
    }

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

