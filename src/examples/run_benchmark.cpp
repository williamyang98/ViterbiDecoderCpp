#include "viterbi/convolutional_encoder.h"
#include "viterbi/convolutional_encoder_lookup.h"
#include "viterbi/convolutional_encoder_shift_register.h"

#include "codes.h"
#include "decoder_factories.h"
#include "decoding_types.h"
#include "decoding_modes.h"
#include "test_helpers.h"
#include "timer.h"
#include "getopt/getopt.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <vector>
#include <random>

constexpr int NOISE_MAX = 100;
enum SelectedMode {
    SOFT16, SOFT8, HARD8
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

template <typename soft_t, typename error_t, class factory_t>
void init_test(
    const Code& code, const DecodeType decode_type,
    const ViterbiDecoder_Config<error_t>& config,
    const soft_t soft_decision_high,
    const soft_t soft_decision_low,
    const uint64_t noise_level, const bool is_soft_noise,
    const size_t total_input_bytes, 
    const size_t total_runs
);

template <typename soft_t, class T>
TestResults run_test(
    T& vitdec, 
    const soft_t* symbols, const size_t total_symbols, 
    const uint8_t* in_bytes, uint8_t* out_bytes, const size_t total_input_bytes,
    const size_t total_runs
);

void usage() {
    fprintf(stderr, 
        " run_benchmark, Runs benchmark on viterbi decoding\n\n"
        "    [-c <code id> (default: 0)]\n"
        "    [-M <mode> (default: soft_16)]\n"
        "        soft_16: use u16 error type and soft decision boundaries\n"
        "        soft_8:  use u8  error type and soft decision boundaries\n"
        "        hard_8:  use u8  error type and hard decision boundaries\n"
        "    [-n <noise level> (default: 0)]\n"
        "    [-s <random seed> (default: Random)]\n"
        "    [-L <total input bytes> (default: 1024)]\n"
        "    [-T <total runs> (default: 1000) ]\n"
        "    [-l Lists all available codes]\n"
        "    [-h Show usage]\n"
    );
}

int main(int argc, char** argv) {
    const size_t N_max = common_codes.size();
    assert(N_max > 0u);

    int config_type = 0;
    int noise_level = 0;
    int random_seed = 0;
    bool is_randomise_seed = true;
    int total_input_bytes = 1024;
    int total_runs = 1000;
    bool is_show_list = false;
    const char* mode_str = NULL;

	int opt; 
    while ((opt = getopt_custom(argc, argv, "c:M:n:s:L:T:lh")) != -1) {
        switch (opt) {
        case 'c':
            config_type = atoi(optarg);
            break;
        case 'M':
            mode_str = optarg;
            break;
        case 'n':
            noise_level = atoi(optarg);
            break;
        case 's':
            is_randomise_seed = false;
            random_seed = atoi(optarg);
            break;
        case 'L':
            total_input_bytes = atoi(optarg);
            break;
        case 'T':
            total_runs = atoi(optarg);
            break;
        case 'l':
            is_show_list = true;
            break;
        case 'h':
        default:
            usage();
            return 0;
        }
    }

    // Update selected decode mode
    auto selected_mode = SelectedMode::SOFT16;
    if (mode_str != NULL) {
        if (strncmp(mode_str, "soft_16", 8) == 0) {
            selected_mode = SelectedMode::SOFT16;
        } else if (strncmp(mode_str, "soft_8", 7) == 0) {
            selected_mode = SelectedMode::SOFT8;
        } else if (strncmp(mode_str, "hard_8", 7) == 0) {
            selected_mode = SelectedMode::HARD8;
        } else {
            fprintf(
                stderr, 
                "Invalid option for mode='%s'\n"
                "Run '%s -h' for description of '-M'\n", 
                mode_str, 
                argv[0]);
            return 1;
        }
    }

    switch (selected_mode) {
    case SelectedMode::SOFT16: printf("Using soft_16 decoders\n"); break;
    case SelectedMode::SOFT8:  printf("Using soft_8 decoders\n"); break;
    case SelectedMode::HARD8:  printf("Using hard_8 decoders\n"); break;
    default:
        break; 
    }

    // Get the simd requirements
    const size_t* K_simd_requirements = NULL;
    switch (selected_mode) {
    case SelectedMode::SOFT16:
        K_simd_requirements = ViterbiDecoder_Factory_u16::K_simd_requirements;
        break; 
    case SelectedMode::SOFT8:
    case SelectedMode::HARD8:
    default:
        K_simd_requirements = ViterbiDecoder_Factory_u8::K_simd_requirements;
        break; 
    }

    // Other arguments
    if (is_show_list) {
        list_codes(common_codes.data(), common_codes.size(), K_simd_requirements);
        return 0;
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

    // NOTE: Hard decision decoding has hard upper limit on noise level
    if (selected_mode == SelectedMode::HARD8) {
        if ((noise_level < 0) || (noise_level > NOISE_MAX)) {
            fprintf(
                stderr,
                "Hard decision noise level must be between %d...%d\n",
                0, NOISE_MAX
            );
            return 1;
        }
    }

    if (total_input_bytes < 0) {
        fprintf(stderr, "Total input bytes must be positive\n");
        return 1;
    }

    if (total_runs < 0) {
        fprintf(stderr, "Total runs must be positive\n");
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

    const auto& code = common_codes[config_type];
    const DecodeType decode_type = get_fastest_simd_type(code.K, K_simd_requirements);

    // Run decoder for selected mode
    if (selected_mode == SelectedMode::SOFT16) {
        auto config = get_soft16_decoding_config(code.R);
        init_test<int16_t, uint16_t, ViterbiDecoder_Factory_u16>(
            code, decode_type,
            config.decoder_config,
            config.soft_decision_high, config.soft_decision_low,
            uint64_t(noise_level), true,
            size_t(total_input_bytes),
            size_t(total_runs)
        );
    } else if (selected_mode == SelectedMode::SOFT8) {
        auto config = get_soft8_decoding_config(code.R);
        init_test<int8_t, uint8_t, ViterbiDecoder_Factory_u8>(
            code, decode_type,
            config.decoder_config,
            config.soft_decision_high, config.soft_decision_low,
            uint64_t(noise_level), true,
            size_t(total_input_bytes),
            size_t(total_runs)
        );
    } else if (selected_mode == SelectedMode::HARD8) {
        auto config = get_hard8_decoding_config(code.R);
        init_test<int8_t, uint8_t, ViterbiDecoder_Factory_u8>(
            code, decode_type,
            config.decoder_config,
            config.soft_decision_high, config.soft_decision_low,
            uint64_t(noise_level), false,
            size_t(total_input_bytes),
            size_t(total_runs)
        );
    } else {
        fprintf(stderr, "Got an invalid decoding mode\n");
    }

    return 0;
}

template <typename soft_t, typename error_t, class factory_t>
void init_test(
    const Code& code, const DecodeType decode_type,
    const ViterbiDecoder_Config<error_t>& config,
    const soft_t soft_decision_high,
    const soft_t soft_decision_low,
    const uint64_t noise_level, const bool is_soft_noise,
    const size_t total_input_bytes, 
    const size_t total_runs
) {
    static_assert(
        sizeof(soft_t) == sizeof(error_t), 
        "Soft decoder requires error and soft types be the same size"
    );

    const auto* name = code.name.c_str();
    const size_t K = code.K;
    const size_t R = code.R;
    const auto* G = code.G.data();
    constexpr size_t K_max_lookup = 10;
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
    size_t simd_alignment = sizeof(soft_t);
    switch (decode_type) {
    case DecodeType::SIMD_SSE: simd_alignment = 16u; break;
    case DecodeType::SIMD_AVX: simd_alignment = 32u; break;
    default:                                         break;
    }
    auto branch_table = ViterbiBranchTable<soft_t>(
        K, R, G, 
        soft_decision_high, soft_decision_low, 
        simd_alignment
    );

    // Generate test data
    std::vector<uint8_t> tx_input_bytes;
    std::vector<soft_t> output_symbols; 
    std::vector<uint8_t> rx_input_bytes;
    tx_input_bytes.resize(total_input_bytes);
    rx_input_bytes.resize(total_input_bytes);
    {
        const size_t total_tail_bits = K-1u;
        const size_t total_data_bits = total_input_bytes*8;
        const size_t total_bits = total_data_bits + total_tail_bits;
        const size_t total_symbols = total_bits * R;
        output_symbols.resize(total_symbols);
    }

    generate_random_bytes(tx_input_bytes.data(), tx_input_bytes.size());
    encode_data(
        enc, 
        tx_input_bytes.data(), tx_input_bytes.size(), 
        output_symbols.data(), output_symbols.size(),
        soft_decision_high, soft_decision_low
    );

    // generate appropriate noise signal
    if (noise_level > 0) {
        if (is_soft_noise) {
            add_noise(output_symbols.data(), output_symbols.size(), noise_level);
            clamp_vector(output_symbols.data(), output_symbols.size(), soft_decision_low, soft_decision_high);
        } else {
            add_binary_noise(output_symbols.data(), output_symbols.size(), noise_level, uint64_t(NOISE_MAX));
        }
    }
    const size_t total_output_symbols = output_symbols.size();
    delete enc;

    // Run tests
    TestResults test_scalar, test_simd_sse, test_simd_avx;
    printf("Starting total_runs=%zu\n", total_runs);
    if (decode_type >= DecodeType::SCALAR) {
        auto vitdec = factory_t::get_scalar(branch_table, config);
        vitdec.set_traceback_length(total_input_bits);
        test_scalar = run_test(
            vitdec, 
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

    if (decode_type >= DecodeType::SIMD_SSE) {
        auto vitdec = factory_t::get_simd_sse(branch_table, config);
        vitdec.set_traceback_length(total_input_bits);
        test_simd_sse = run_test(
            vitdec, 
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

    if (decode_type >= DecodeType::SIMD_AVX) {
        auto vitdec = factory_t::get_simd_avx(branch_table, config);
        vitdec.set_traceback_length(total_input_bits);
        test_simd_avx = run_test(
            vitdec, 
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

template <typename soft_t, class T>
TestResults run_test(
    T& vitdec, 
    const soft_t* symbols, const size_t total_symbols, 
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
            vitdec.reset();
            results.us_reset += float(t.get_delta());
        }
        {
            Timer t;
            vitdec.update(symbols, total_symbols);
            results.us_update += float(t.get_delta());
        }
        {
            Timer t;
            const uint64_t error = vitdec.chainback(out_bytes, total_input_bits, 0u);
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