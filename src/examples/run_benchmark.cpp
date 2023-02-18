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

class Timer 
{
private:
    std::chrono::steady_clock::time_point dt_start;
public:
    Timer() {
        dt_start = std::chrono::high_resolution_clock::now();
    } 

    template <typename T = std::chrono::microseconds>
    uint64_t get_delta() {
        const auto dt_end = std::chrono::high_resolution_clock::now();
        const auto dt_delta = dt_end - dt_start;
        return std::chrono::duration_cast<T>(dt_delta).count();
    }
};

struct TestResults {
    float us_reset = 0.0f;      // microseconds
    float us_update = 0.0f;
    float us_chainback = 0.0f;
    float bit_error_rate = 0.0f;      
    size_t total_incorrect_bits = 0u;
    size_t total_decoded_bits = 0u;
    size_t total_error = 0u;
    size_t total_runs = 0u;
};

void init_test(
    const Code& code, 
    const int noise_level, 
    const size_t total_input_bytes, 
    const size_t total_runs
);

TestResults run_test(
    ViterbiDecoder<int16_t>* vitdec, 
    const int16_t* symbols, const size_t total_symbols, 
    const uint8_t* in_bytes, uint8_t* out_bytes, const size_t total_input_bytes,
    const size_t total_runs
);

void usage() {
    fprintf(stderr, 
        " run_benchmark, Runs benchmark on viterbi decoding using 16bit soft decision values\n\n"
        "    [-c <code id> (default: 0)]\n"
        "    [-n <noise level> (default: 0) ]\n"
        "    [-s <random seed> (default: 0) ]\n"
        "    [-S Randomises seed ]\n"
        "    [-L <total input bytes> (default: 1024) ]\n"
        "    [-T <total runs> (default: 10000) ]\n"
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
    int total_runs = 10000;

	int opt; 
    while ((opt = getopt_custom(argc, argv, "c:n:s:SL:T:lh")) != -1) {
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
        case 'T':
            total_runs = atoi(optarg);
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

    if (noise_level < 0) {
        fprintf(stderr, "Noise level must be positive\n");
        return 1;
    }

    if (total_input_bytes < 0) {
        fprintf(stderr, "Total input bytes must be positive\n");
        return 1;
    }

    if (total_runs < 0) {
        fprintf(stderr, "Total runs must be positive\n");
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
    init_test(code, noise_level, size_t(total_input_bytes), size_t(total_runs));
    
    return 0;
}

void init_test(
    const Code& code, 
    const int noise_level, 
    const size_t total_input_bytes, 
    const size_t total_runs)
{
    const auto* name = code.name.c_str();
    const size_t K = code.K;
    const size_t R = code.R;
    const auto* G = code.G.data();
    constexpr size_t K_max_lookup = 10;
    constexpr int16_t SOFT_DECISION_HIGH = +127;
    constexpr int16_t SOFT_DECISION_LOW  = -127;
    const size_t total_input_bits = total_input_bytes*8u;

    printf("Using '%s': K=%zu, R=%zu\n", name, K, R);

    // Decide suitable convolutional encoder
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

    // Create branch table with correct alignment 
    size_t simd_alignment = sizeof(int16_t);
    switch (code.decode_type) {
    case DecodeType::SIMD_SSE: simd_alignment = 16u; break;
    case DecodeType::SIMD_AVX: simd_alignment = 32u; break;
    default:                                         break;
    }
    auto branch_table = ViterbiBranchTable<int16_t>(K, R, G, SOFT_DECISION_HIGH, SOFT_DECISION_LOW, simd_alignment);

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
    const size_t total_output_symbols = output_symbols.size();
    delete enc;

    // Run tests
    TestResults test_scalar, test_simd_sse, test_simd_avx;
    printf("Starting total_runs=%zu\n", total_runs);
    if (code.decode_type >= DecodeType::SCALAR) {
        auto vitdec = ViterbiDecoder_Scalar<uint16_t, int16_t>(branch_table);
        vitdec.set_traceback_length(total_input_bits);
        test_scalar = run_test(
            &vitdec, 
            output_symbols.data(), output_symbols.size(), 
            tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes,
            total_runs
        );

        printf("> Scalar results\n");
        printf("us_reset     = %.2f\n", test_scalar.us_reset);
        printf("us_update    = %.2f\n", test_scalar.us_update);
        printf("us_chainback = %.2f\n", test_scalar.us_chainback);
        printf("ber          = %.4f\n", test_scalar.bit_error_rate);
        printf("errors       = %zu/%zu\n", test_scalar.total_incorrect_bits, test_scalar.total_decoded_bits);
        printf("error_metric = %zu\n",  test_scalar.total_error);
        printf("\n");
    }

    if (code.decode_type >= DecodeType::SIMD_SSE) {
        auto vitdec = ViterbiDecoder_SSE(branch_table);
        vitdec.set_traceback_length(total_input_bits);
        test_simd_sse = run_test(
            &vitdec, 
            output_symbols.data(), output_symbols.size(), 
            tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes,
            total_runs
        );

        printf("> SIMD_SSE results\n");
        printf("us_reset     = %.2f (x%.2f)\n", test_simd_sse.us_reset, test_scalar.us_reset/test_simd_sse.us_reset);
        printf("us_update    = %.2f (x%.2f)\n", test_simd_sse.us_update, test_scalar.us_update/test_simd_sse.us_update);
        printf("us_chainback = %.2f (x%.2f)\n", test_simd_sse.us_chainback, test_scalar.us_chainback/test_simd_sse.us_chainback);
        printf("ber          = %.4f\n", test_simd_sse.bit_error_rate);
        printf("errors       = %zu/%zu\n", test_simd_sse.total_incorrect_bits, test_simd_sse.total_decoded_bits);
        printf("error_metric = %zu\n",  test_simd_sse.total_error);
        printf("\n");
    }

    if (code.decode_type >= DecodeType::SIMD_AVX) {
        auto vitdec = ViterbiDecoder_AVX(branch_table);
        vitdec.set_traceback_length(total_input_bits);
        test_simd_avx = run_test(
            &vitdec, 
            output_symbols.data(), output_symbols.size(), 
            tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes,
            total_runs
        );

        printf("> SIMD_AVX results\n");
        printf("us_reset     = %.2f (x%.2f)\n", test_simd_avx.us_reset, test_scalar.us_reset/test_simd_avx.us_reset);
        printf("us_update    = %.2f (x%.2f)\n", test_simd_avx.us_update, test_scalar.us_update/test_simd_avx.us_update);
        printf("us_chainback = %.2f (x%.2f)\n", test_simd_avx.us_chainback, test_scalar.us_chainback/test_simd_avx.us_chainback);
        printf("ber          = %.4f\n", test_simd_avx.bit_error_rate);
        printf("errors       = %zu/%zu\n", test_simd_avx.total_incorrect_bits, test_simd_avx.total_decoded_bits);
        printf("error_metric = %zu\n",  test_simd_avx.total_error);
        printf("\n");
    }
}

TestResults run_test(
    ViterbiDecoder<int16_t>* vitdec, 
    const int16_t* symbols, const size_t total_symbols, 
    const uint8_t* in_bytes, uint8_t* out_bytes, const size_t total_input_bytes,
    const size_t total_runs
) {
    const size_t total_input_bits = total_input_bytes*8u;
    constexpr size_t print_rate = 1u;

    TestResults results;
    for (size_t curr_run = 0u; curr_run < total_runs; curr_run++) {
        if (curr_run % print_rate == 0) {
            printf("Run: %zu/%zu\r", curr_run, total_runs);
        }
        {
            Timer t;
            vitdec->reset();
            results.us_reset += float(t.get_delta());
        }
        {
            Timer t;
            vitdec->update(symbols, total_symbols);
            results.us_update += float(t.get_delta());
        }
        {
            Timer t;
            const uint64_t error = vitdec->chainback(out_bytes, total_input_bits, 0u);
            results.total_error += error;
            results.us_chainback += float(t.get_delta());
        }
        const size_t total_errors = get_total_bit_errors(in_bytes, out_bytes, total_input_bytes);
        results.total_incorrect_bits += total_errors;
        results.total_decoded_bits += total_input_bits;
        results.total_runs++;
    }
    printf("%*s\r", 100, "");

    results.bit_error_rate = float(results.total_incorrect_bits) / float(results.total_decoded_bits);
    return results;
}