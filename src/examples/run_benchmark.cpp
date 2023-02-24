#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <inttypes.h>
#include <vector>
#include <random>

#include "viterbi/convolutional_encoder.h"
#include "viterbi/convolutional_encoder_lookup.h"
#include "viterbi/convolutional_encoder_shift_register.h"

#include "common_codes.h"
#include "decoder_factories.h"
#include "decoding_modes.h"
#include "test_helpers.h"
#include "timer.h"
#include "getopt/getopt.h"

constexpr int NOISE_MAX = 100;
enum SelectedMode {
    SOFT16, SOFT8, HARD8
};

struct TestResults {
    float ns_reset = 0.0f;      // microseconds
    float ns_update = 0.0f;
    float ns_chainback = 0.0f;
    float bit_error_rate = 0.0f;      
    size_t total_incorrect_bits = 0u;
    size_t total_decoded_bits = 0u;
    size_t total_error = 0u;
    size_t total_runs = 0u;
};

template <template <size_t, size_t> class factory_t, typename ... U>
void select_test(
    const size_t code_id,
    U&& ... args
);

template <template <size_t, size_t> class factory_t, size_t K, size_t R, typename soft_t, typename error_t>
void init_test(
    const Code<K,R>& code, 
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
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
    const size_t N_max = common_codes.N;
    assert(N_max > 0u);

    int code_id = 0;
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
            code_id = atoi(optarg);
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

    // Other arguments
    if (is_show_list) {
        switch (selected_mode) {
        case SelectedMode::SOFT16:
            list_codes<ViterbiDecoder_Factory_u16>();
            break; 
        case SelectedMode::SOFT8:
        case SelectedMode::HARD8:
            list_codes<ViterbiDecoder_Factory_u8>();
            break;
        default:
            break; 
        }
        return 0;
    }

    if ((code_id < 0) || (code_id >= (int)N_max)) {
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

     // Select code
    switch (selected_mode) {
    case SelectedMode::SOFT16:
        select_test<ViterbiDecoder_Factory_u16>(
            size_t(code_id), 
            get_soft16_decoding_config,
            uint64_t(noise_level), true, 
            size_t(total_input_bytes),
            size_t(total_runs)
        );
        break; 
    case SelectedMode::SOFT8:
        select_test<ViterbiDecoder_Factory_u8>(
            size_t(code_id), 
            get_soft8_decoding_config,
            uint64_t(noise_level), true, 
            size_t(total_input_bytes),
            size_t(total_runs)
        );
        break;
    case SelectedMode::HARD8:
        select_test<ViterbiDecoder_Factory_u8>(
            size_t(code_id), 
            get_hard8_decoding_config,
            uint64_t(noise_level), false, 
            size_t(total_input_bytes),
            size_t(total_runs)
        );
        break;
    default:
        break; 
    }

    return 0;
}

template <template <size_t, size_t> class factory_t, typename ... U>
void select_test(
    const size_t code_id,
    U&& ... args
) {
    switch (code_id) {
    case 0: return init_test<factory_t>(common_codes.code_0, std::forward<U>(args)...);
    case 1: return init_test<factory_t>(common_codes.code_1, std::forward<U>(args)...);
    case 2: return init_test<factory_t>(common_codes.code_2, std::forward<U>(args)...);
    case 3: return init_test<factory_t>(common_codes.code_3, std::forward<U>(args)...);
    case 4: return init_test<factory_t>(common_codes.code_4, std::forward<U>(args)...);
    case 5: return init_test<factory_t>(common_codes.code_5, std::forward<U>(args)...);
    case 6: return init_test<factory_t>(common_codes.code_6, std::forward<U>(args)...);
    case 7: return init_test<factory_t>(common_codes.code_7, std::forward<U>(args)...);
    }
}

template <template <size_t, size_t> class factory_t, size_t K, size_t R, typename soft_t, typename error_t>
void init_test(
    const Code<K,R>& code, 
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    const uint64_t noise_level, const bool is_soft_noise,
    const size_t total_input_bytes,
    const size_t total_runs 
) {

    printf("Using '%s': K=%zu, R=%zu\n", code.name, code.K, code.R);

    const Decoder_Config<soft_t, error_t> config = config_factory(code.R);
    auto enc = ConvolutionalEncoder_ShiftRegister(code.K, code.R, code.G.data());
    auto branch_table = ViterbiBranchTable<K,R,soft_t>(code.G.data(), config.soft_decision_high, config.soft_decision_low);

    // Generate test data
    const size_t total_input_bits = total_input_bytes*8u;
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
        &enc, 
        tx_input_bytes.data(), tx_input_bytes.size(), 
        output_symbols.data(), output_symbols.size(),
        config.soft_decision_high, config.soft_decision_low
    );

    // generate appropriate noise signal
    if (noise_level > 0) {
        if (is_soft_noise) {
            add_noise(output_symbols.data(), output_symbols.size(), noise_level);
            clamp_vector(output_symbols.data(), output_symbols.size(), config.soft_decision_low, config.soft_decision_high);
        } else {
            add_binary_noise(output_symbols.data(), output_symbols.size(), noise_level, uint64_t(NOISE_MAX));
        }
    }

    // compare results
    auto print_independent = [](const TestResults src) {
        constexpr float ns_to_sec = 1e-9f;
        // printf("sec_reset     = %.3f\n", src.us_reset * ns_to_sec);
        printf("sec_update    = %.3f\n", src.ns_update * ns_to_sec);
        // printf("sec_chainback = %.3f\n", src.ns_chainback * ns_to_sec);
        printf("ber           = %.4f\n", src.bit_error_rate);
        printf("errors        = %zu/%zu\n", src.total_incorrect_bits, src.total_decoded_bits);
        printf("error_metric  = %" PRIu64 "\n",  src.total_error);
    };

    auto print_comparison = [](const TestResults ref, const TestResults src) {
        constexpr float ns_to_sec = 1e-9f;
        // printf("sec_reset     = %.3f (x%.2f)\n", src.ns_reset * ns_to_sec, ref.ns_reset/src.ns_reset);
        printf("sec_update    = %.3f (x%.2f)\n", src.ns_update * ns_to_sec, ref.ns_update/src.ns_update);
        // printf("sec_chainback = %.3f (x%.2f)\n", src.ns_chainback * ns_to_sec, ref.ns_chainback/src.ns_chainback);
        printf("ber           = %.4f\n", src.bit_error_rate);
        printf("errors        = %zu/%zu\n", src.total_incorrect_bits, src.total_decoded_bits);
        printf("error_metric  = %" PRIu64 "\n",  src.total_error);
    };

    // Run tests
    TestResults test_ref;
    printf("Starting total_runs=%zu\n", total_runs);
    if constexpr(factory_t<K,R>::Scalar::is_valid) {
        auto vitdec = typename factory_t<K,R>::Scalar(branch_table, config.decoder_config);
        vitdec.set_traceback_length(total_input_bits);
        const auto res = run_test(
            vitdec, 
            output_symbols.data(), output_symbols.size(), 
            tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes,
            total_runs
        );

        printf("> Scalar results\n");
        print_independent(res);
        test_ref = res;
        printf("\n");
    }

    if constexpr(factory_t<K,R>::SIMD_SSE::is_valid) {
        auto vitdec = typename factory_t<K,R>::SIMD_SSE(branch_table, config.decoder_config);
        vitdec.set_traceback_length(total_input_bits);
        const auto res = run_test(
            vitdec, 
            output_symbols.data(), output_symbols.size(), 
            tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes,
            total_runs
        );

        printf("> SIMD_SSE results\n");
        print_comparison(test_ref, res);
        printf("\n");
    }

    if constexpr(factory_t<K,R>::SIMD_AVX::is_valid) {
        auto vitdec = typename factory_t<K,R>::SIMD_AVX(branch_table, config.decoder_config);
        vitdec.set_traceback_length(total_input_bits);
        const auto res = run_test(
            vitdec, 
            output_symbols.data(), output_symbols.size(), 
            tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes,
            total_runs
        );

        printf("> SIMD_AVX results\n");
        print_comparison(test_ref, res);
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
            results.ns_reset += float(t.get_delta());
        }
        {
            Timer t;
            vitdec.update(symbols, total_symbols);
            results.ns_update += float(t.get_delta());
        }
        {
            Timer t;
            vitdec.chainback(out_bytes, total_input_bits, 0u);
            results.ns_chainback += float(t.get_delta());
        }
        {
            const uint64_t error = vitdec.get_error();
            results.total_error += error;
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