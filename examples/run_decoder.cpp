#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <inttypes.h>
#include <vector>
#include <random>
#include <chrono>

#include "convolutional_encoder.h"
#include "convolutional_encoder_shift_register.h"

#include "helpers/common_codes.h"
#include "helpers/decoder_configs.h"
#include "helpers/decoder_factories.h"
#include "helpers/test_helpers.h"
#include "getopt/getopt.h"

constexpr int NOISE_MAX = 100;
enum DecodeType {
    SOFT16, SOFT8, HARD8
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
    const SIMD_Type decode_type,
    const uint64_t noise_level, const bool is_soft_noise,
    const size_t total_input_bytes 
);

template <typename soft_t, class T>
void run_test(
    T& vitdec, 
    ConvolutionalEncoder* enc, 
    const uint64_t noise_level,
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
        #if defined(VITERBI_SIMD_X86)
        "        sse:    128bit vectorisation\n"
        "        avx:    256bit vectorisation\n"
        #elif defined(VITERBI_SIMD_ARM)
        "        neon:   128bit vectorisation\n"
        #endif
        "    [-n <noise level> (default: 0)]\n"
        "    [-s <random seed> (default: Random)]\n"
        "    [-L <total input bytes> (default: 1024)]\n"
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
    int total_input_bytes = 1024;
    bool is_randomise_seed = true;
    bool is_show_list = false;
    const char* mode_str = NULL;
    const char* decode_type_str = NULL;

	int opt; 
    while ((opt = getopt_custom(argc, argv, "c:M:d:n:s:L:lh")) != -1) {
        switch (opt) {
        case 'c':
            code_id = atoi(optarg);
            break;
        case 'M':
            mode_str = optarg;
            break;
        case 'd':
            decode_type_str = optarg;
            break;
        case 'n':
            is_randomise_seed = false;
            noise_level = atoi(optarg);
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
    auto selected_mode = DecodeType::SOFT16;
    if (mode_str != NULL) {
        if (strncmp(mode_str, "soft_16", 8) == 0) {
            selected_mode = DecodeType::SOFT16;
        } else if (strncmp(mode_str, "soft_8", 7) == 0) {
            selected_mode = DecodeType::SOFT8;
        } else if (strncmp(mode_str, "hard_8", 7) == 0) {
            selected_mode = DecodeType::HARD8;
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
    case DecodeType::SOFT16: printf("Using soft_16 decoders\n"); break;
    case DecodeType::SOFT8:  printf("Using soft_8 decoders\n"); break;
    case DecodeType::HARD8:  printf("Using hard_8 decoders\n"); break;
    default:
        break; 
    }

    // Other arguments
    if (is_show_list) {
        switch (selected_mode) {
        case DecodeType::SOFT16:
            list_codes<ViterbiDecoder_Factory_u16>();
            break; 
        case DecodeType::SOFT8:
        case DecodeType::HARD8:
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
        fprintf(
            stderr,
            "Noise level must be positive\n"
        );
        return 1;
    }

    // NOTE: Hard decision decoding has hard upper limit on noise level
    if (selected_mode == DecodeType::HARD8) {
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

    auto decode_type = SIMD_Type::SCALAR;
    #if defined(VITERBI_SIMD_X86)
    decode_type = SIMD_Type::SIMD_AVX;
    #elif defined(VITERBI_SIMD_ARM)
    decode_type = SIMD_Type::SIMD_NEON;
    #endif

    if (decode_type_str != NULL) {
        if (strncmp(decode_type_str, "scalar", 7) == 0) {
            decode_type = SIMD_Type::SCALAR;
        #if defined(VITERBI_SIMD_X86)
        } else if (strncmp(decode_type_str, "sse", 4) == 0) {
            decode_type = SIMD_Type::SIMD_SSE;
        } else if (strncmp(decode_type_str, "avx", 4) == 0) {
            decode_type = SIMD_Type::SIMD_AVX;
        #elif defined(VITERBI_SIMD_ARM)
        } else if (strncmp(decode_type_str, "neon", 5) == 0) {
            decode_type = SIMD_Type::SIMD_NEON;
        #endif
        } else {
            fprintf(
                stderr, 
                "Invalid option for decode_type='%s'\n"
                "Run '%s -h' for description of '-d'\n", 
                decode_type_str, 
                argv[0]);
            return 1;
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

    // Select code
    switch (selected_mode) {
    case DecodeType::SOFT16:
        select_test<ViterbiDecoder_Factory_u16>(
            size_t(code_id), 
            get_soft16_decoding_config,
            decode_type,
            uint64_t(noise_level), true, 
            size_t(total_input_bytes)
        );
        break; 
    case DecodeType::SOFT8:
        select_test<ViterbiDecoder_Factory_u8>(
            size_t(code_id), 
            get_soft8_decoding_config,
            decode_type,
            uint64_t(noise_level), true, 
            size_t(total_input_bytes)
        );
        break;
    case DecodeType::HARD8:
        select_test<ViterbiDecoder_Factory_u8>(
            size_t(code_id), 
            get_hard8_decoding_config,
            decode_type,
            uint64_t(noise_level), false, 
            size_t(total_input_bytes)
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
    const SIMD_Type decode_type,
    const uint64_t noise_level, const bool is_soft_noise,
    const size_t total_input_bytes 
) {
    printf("Using '%s': K=%zu, R=%zu\n", code.name, code.K, code.R);

    const Decoder_Config<soft_t, error_t> config = config_factory(code.R);
    auto enc = ConvolutionalEncoder_ShiftRegister(code.K, code.R, code.G.data());
    auto branch_table = ViterbiBranchTable<K,R,soft_t>(code.G.data(), config.soft_decision_high, config.soft_decision_low);

    // Run decoder
    #if defined(VITERBI_SIMD_X86)
    if (decode_type >= SIMD_Type::SIMD_AVX) {
        if constexpr(factory_t<K,R>::SIMD_AVX::is_valid) {
            printf("Using SIMD_AVX decoder\n");
            auto vitdec = typename factory_t<K,R>::SIMD_AVX(branch_table, config.decoder_config);
            run_test(
                vitdec, &enc, 
                noise_level, is_soft_noise, 
                total_input_bytes, 
                config.soft_decision_high, config.soft_decision_low
            );
            return;
        } else {
            printf("Requested SIMD_AVX decoder unsuccessfully\n");
        }
    }

    if (decode_type >= SIMD_Type::SIMD_SSE) {
        if constexpr(factory_t<K,R>::SIMD_SSE::is_valid) {
            printf("Using SIMD_SSE decoder\n");
            auto vitdec = typename factory_t<K,R>::SIMD_SSE(branch_table, config.decoder_config);
            run_test(
                vitdec, &enc, 
                noise_level, is_soft_noise, 
                total_input_bytes, 
                config.soft_decision_high, config.soft_decision_low
            );
            return;
        } else {
            printf("Requested SIMD_SSE decoder unsuccessfully\n");
        }
    }
    #elif defined(VITERBI_SIMD_ARM)
    if (decode_type >= SIMD_Type::SIMD_NEON) {
        if constexpr(factory_t<K,R>::SIMD_NEON::is_valid) {
            printf("Using SIMD_NEON decoder\n");
            auto vitdec = typename factory_t<K,R>::SIMD_NEON(branch_table, config.decoder_config);
            run_test(
                vitdec, &enc, 
                noise_level, is_soft_noise, 
                total_input_bytes, 
                config.soft_decision_high, config.soft_decision_low
            );
            return;
        } else {
            printf("Requested SIMD_NEON decoder unsuccessfully\n");
        }
    }
    #endif
    if (decode_type >= SIMD_Type::SCALAR) {
        if constexpr(factory_t<K,R>::Scalar::is_valid) {
            printf("Using SCALAR decoder\n");
            auto vitdec = typename factory_t<K,R>::Scalar(branch_table, config.decoder_config);
            run_test(
                vitdec, &enc, 
                noise_level, is_soft_noise, 
                total_input_bytes, 
                config.soft_decision_high, config.soft_decision_low
            );
            return;
        } else {
            printf("Requested SCALAR decoder unsuccessfully\n");
        }
    }
}

template <typename soft_t, class T>
void run_test(
    T& vitdec, 
    ConvolutionalEncoder* enc, 
    const uint64_t noise_level,
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
    vitdec.chainback(rx_input_bytes.data(), total_input_bits, 0u);
    const uint64_t error = vitdec.get_error();
    printf("error=%" PRIu64 "\n", error);

    // Show decoding results
    const size_t total_errors = get_total_bit_errors(tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes);
    const float bit_error_rate = (float)total_errors / (float)total_input_bits * 100.0f;
    printf("bit error rate=%.2f%%\n", bit_error_rate);
    printf("%zu/%zu incorrect bits\n", total_errors, total_input_bits);
}
