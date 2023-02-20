#include "viterbi/convolutional_encoder.h"
#include "viterbi/convolutional_encoder_lookup.h"
#include "viterbi/convolutional_encoder_shift_register.h"

#include "codes.h"
#include "decoder_factories.h"
#include "test_helpers.h"
#include "getopt/getopt.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <vector>
#include <random>
#include <chrono>

constexpr int NOISE_MAX = 100;
enum SelectedMode {
    SOFT16, SOFT8, HARD8
};

template <typename soft_t, typename error_t, class factory_t>
void init_test(
    const Code& code, DecodeType decode_type,
    const ViterbiDecoder_Config<error_t>& config,
    const soft_t soft_decision_high,
    const soft_t soft_decision_low,
    const size_t noise_level, const bool is_soft_noise,
    const size_t total_input_bytes 
);

template <typename soft_t, class T>
void run_test(
    T& vitdec, 
    ConvolutionalEncoder* enc, 
    const size_t noise_level,
    const bool is_soft_noise,
    const size_t total_input_bytes,
    const soft_t soft_decision_high,
    const soft_t soft_decision_low
);

void usage() {
    fprintf(stderr, 
        "run_decoder, Runs viterbi decoder\n\n"
        "    [-c <code id> (default: 0)]\n"
        "    [-M <mode> (default: soft_16)]\n"
        "        soft_16: use u16 error type and soft decision boundaries\n"
        "        soft_8:  use u8  error type and soft decision boundaries\n"
        "        hard_8:  use u8  error type and hard decision boundaries\n"
        "    [-d <decode type> (default: highest)]\n"
        "        scalar:     no vectorisation\n"
        "        sse:    128bit vectorisation\n"
        "        avx:    256bit vectorisation\n"
        "    [-n <noise level> (default: 0)]\n"
        "    [-s <random seed> (default: 0)]\n"
        "    [-S Randomises seed ]\n"
        "    [-L <total input bytes> (default: 1024)]\n"
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
    int total_input_bytes = 1024;
    bool is_randomise_seed = false;
    bool is_show_list = false;
    const char* mode_str = NULL;
    const char* decode_type_str = NULL;

	int opt; 
    while ((opt = getopt_custom(argc, argv, "c:M:d:n:s:SL:lh")) != -1) {
        switch (opt) {
        case 'c':
            config_type = atoi(optarg);
            break;
        case 'M':
            mode_str = optarg;
            break;
        case 'd':
            decode_type_str = optarg;
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
        fprintf(
            stderr,
            "Noise level must be positive\n"
        );
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


    // Get valid decode type for selected code
    const auto& code = common_codes[config_type];

    DecodeType decode_type = get_fastest_simd_type(code.K, K_simd_requirements);
    if (decode_type_str != NULL) {
        auto new_decode_type = DecodeType::SIMD_AVX;
        if (strncmp(decode_type_str, "scalar", 7) == 0) {
            new_decode_type = DecodeType::SCALAR;
        } else if (strncmp(decode_type_str, "sse", 4) == 0) {
            new_decode_type = DecodeType::SIMD_SSE;
        } else if (strncmp(decode_type_str, "avx", 4) == 0) {
            new_decode_type = DecodeType::SIMD_AVX;
        } else {
            fprintf(
                stderr, 
                "Invalid option for decode_type='%s'\n"
                "Run '%s -h' for description of '-d'\n", 
                decode_type_str, argv[0]);
            return 1;
        }

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

    // Generate seed
    if (is_randomise_seed) {
        const auto dt_now = std::chrono::system_clock::now().time_since_epoch();
        const auto us_now = std::chrono::duration_cast<std::chrono::microseconds>(dt_now).count();
        std::srand((unsigned int)us_now);
        random_seed = std::rand();
        printf("Using random_seed=%d\n", random_seed);
    }
    std::srand((unsigned int)random_seed);

    // Run decoder for selected mode
    if (selected_mode == SelectedMode::SOFT16) {
        const int16_t soft_decision_high = +127;
        const int16_t soft_decision_low  = -127;
        const uint16_t max_error = uint16_t(soft_decision_high-soft_decision_low) * uint16_t(code.R);
        const uint16_t error_margin = max_error * uint16_t(4u);

        ViterbiDecoder_Config<uint16_t> config;
        config.soft_decision_max_error = max_error;
        config.initial_start_error = std::numeric_limits<uint16_t>::min();
        config.initial_non_start_error = config.initial_start_error + error_margin;
        config.renormalisation_threshold = std::numeric_limits<uint16_t>::max() - error_margin;

        init_test<int16_t, uint16_t, ViterbiDecoder_Factory_u16>(
            code, decode_type,
            config,
            soft_decision_high, soft_decision_low,
            size_t(noise_level), true,
            size_t(total_input_bytes)
        );
    } else if (selected_mode == SelectedMode::SOFT8) {
        const int8_t soft_decision_high = +5;
        const int8_t soft_decision_low  = -5;
        const uint8_t max_error = uint8_t(soft_decision_high-soft_decision_low) * uint8_t(code.R);
        const uint8_t error_margin = max_error * uint8_t(1u);

        ViterbiDecoder_Config<uint8_t> config;
        config.soft_decision_max_error = max_error;
        config.initial_start_error = std::numeric_limits<uint8_t>::min();
        config.initial_non_start_error = config.initial_start_error + error_margin;
        config.renormalisation_threshold = std::numeric_limits<uint8_t>::max() - error_margin;

        init_test<int8_t, uint8_t, ViterbiDecoder_Factory_u8>(
            code, decode_type,
            config,
            soft_decision_high, soft_decision_low,
            size_t(noise_level), true,
            size_t(total_input_bytes)
        );
    } else {
        const int8_t soft_decision_high = +1;
        const int8_t soft_decision_low  = -1;
        const uint8_t max_error = uint8_t(soft_decision_high-soft_decision_low) * uint8_t(code.R);
        const uint8_t error_margin = max_error * uint8_t(4u);

        ViterbiDecoder_Config<uint8_t> config;
        config.soft_decision_max_error = max_error;
        config.initial_start_error = std::numeric_limits<uint8_t>::min();
        config.initial_non_start_error = config.initial_start_error + error_margin;
        config.renormalisation_threshold = std::numeric_limits<uint8_t>::max() - error_margin;

        init_test<int8_t, uint8_t, ViterbiDecoder_Factory_u8>(
            code, decode_type,
            config,
            soft_decision_high, soft_decision_low,
            size_t(noise_level), false,
            size_t(total_input_bytes)
        );
    }

    return 0;
}


template <typename soft_t, typename error_t, class factory_t>
void init_test(
    const Code& code, DecodeType decode_type,
    const ViterbiDecoder_Config<error_t>& config,
    const soft_t soft_decision_high,
    const soft_t soft_decision_low,
    const size_t noise_level, const bool is_soft_noise,
    const size_t total_input_bytes 
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

    printf("Using '%s': K=%zu, R=%zu\n", name, K, R);

    // Decide appropriate convolutional encoder
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

    // Create viterbi decoder
    switch (decode_type) {
    case DecodeType::SCALAR:
        {
            printf("Using SCALAR decoder\n");
            auto vitdec = factory_t::get_scalar(branch_table, config);
            run_test(
                vitdec, enc, 
                noise_level, is_soft_noise, 
                total_input_bytes, 
                soft_decision_high, soft_decision_low
            );
        }
        break;
    case DecodeType::SIMD_SSE:
        {
            printf("Using SIMD_SSE with alignment=%zu\n", simd_alignment);
            auto vitdec = factory_t::get_simd_sse(branch_table, config);
            run_test(
                vitdec, enc, 
                noise_level, is_soft_noise, 
                total_input_bytes, 
                soft_decision_high, soft_decision_low
            );
        }
        break;
    case DecodeType::SIMD_AVX:
        {
            printf("Using SIMD_AVX with alignment=%zu\n", simd_alignment);
            auto vitdec = factory_t::get_simd_avx(branch_table, config);
            run_test(
                vitdec, enc, 
                noise_level, is_soft_noise, 
                total_input_bytes, 
                soft_decision_high, soft_decision_low
            );
        }
        break;
    default:
        printf("Unknown decoder type\n");
        exit(1);
        break;
    }

    delete enc;
}

template <typename soft_t, class T>
void run_test(
    T& vitdec, 
    ConvolutionalEncoder* enc, 
    const size_t noise_level,
    const bool is_soft_noise,
    const size_t total_input_bytes,
    const soft_t soft_decision_high,
    const soft_t soft_decision_low
) {
    assert(vitdec.K == enc->K);
    assert(vitdec.R == enc->R);
    const size_t K = vitdec.K; 
    const size_t R = vitdec.R; 
    const size_t total_input_bits = total_input_bytes*8u;
    vitdec.set_traceback_length(total_input_bits);

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
    vitdec.reset();
    vitdec.update(output_symbols.data(), total_output_symbols);
    const uint64_t error = vitdec.chainback(rx_input_bytes.data(), total_input_bits, 0u);
    printf("error=%zu\n", error);

    // Show decoding results
    const size_t total_errors = get_total_bit_errors(tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes);
    const float bit_error_rate = (float)total_errors / (float)total_input_bits * 100.0f;
    printf("bit error rate=%.2f%%\n", bit_error_rate);
    printf("%zu/%zu incorrect bits\n", total_errors, total_input_bits);
}
