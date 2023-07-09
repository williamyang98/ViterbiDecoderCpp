#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <inttypes.h>
#include <vector>
#include <random>
#include <chrono>

#include "viterbi/convolutional_encoder.h"
#include "viterbi/convolutional_encoder_shift_register.h"

#include "helpers/common_codes.h"
#include "helpers/decode_type.h"
#include "helpers/simd_type.h"
#include "helpers/test_helpers.h"
#include "utility/expected.hpp"
#include "getopt/getopt.h"

constexpr int NOISE_MAX = 100;

struct Arguments {
    size_t code_id;
    SIMD_Type simd_type;
    DecodeType decode_type;
    uint64_t noise_level;
    bool is_soft_noise;
    size_t total_input_bytes;
};

template <size_t K, size_t R, typename code_t>
void select_code(const Code<K,R,code_t>& code, Arguments args);

template <template <size_t, size_t> class factory_t, size_t K, size_t R, typename code_t, typename soft_t, typename error_t>
void init_test(
    const Code<K,R,code_t>& code, 
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    Arguments args
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
        "    [-d <decode_type> (default: soft_16)]\n"
        "        soft_16: use u16 error type and soft decision boundaries\n"
        "        soft_8:  use u8  error type and soft decision boundaries\n"
        "        hard_8:  use u8  error type and hard decision boundaries\n"
        "    [-v <simd_type> (default: highest)]\n"
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

tl::expected<Arguments, int> parse_args(int argc, char** argv) {
    struct {
        int code_id = 0;
        int noise_level = 0;
        int random_seed = 0;
        int total_input_bytes = 1024;
        bool is_randomise_seed = true;
        bool is_show_list = false;
        const char* decode_type_str = NULL;
        const char* simd_type_str = NULL;
    } args;

	int opt; 
    while ((opt = getopt_custom(argc, argv, "c:d:v:n:s:L:lh")) != -1) {
        switch (opt) {
        case 'c':
            args.code_id = atoi(optarg);
            break;
        case 'd':
            args.decode_type_str = optarg;
            break;
        case 'v':
            args.simd_type_str = optarg;
            break;
        case 'n':
            args.noise_level = atoi(optarg);
            break;
        case 's':
            args.is_randomise_seed = false;
            args.random_seed = atoi(optarg);
            break;
        case 'L':
            args.total_input_bytes = atoi(optarg);
            break;
        case 'l':
            args.is_show_list = true;
            break;
        case 'h':
        default:
            usage();
            return tl::unexpected(0);
        }
    }

    // Update selected decode mode
    auto decode_type = DecodeType::SOFT16;
    if (args.decode_type_str != NULL) {
        if (strncmp(args.decode_type_str, "soft_16", 8) == 0) {
            decode_type = DecodeType::SOFT16;
        } else if (strncmp(args.decode_type_str, "soft_8", 7) == 0) {
            decode_type = DecodeType::SOFT8;
        } else if (strncmp(args.decode_type_str, "hard_8", 7) == 0) {
            decode_type = DecodeType::HARD8;
        } else {
            fprintf(
                stderr, 
                "Invalid option for mode='%s'\n"
                "Run '%s -h' for description of '-M'\n", 
                args.decode_type_str, 
                argv[0]);
            return tl::unexpected(1);
        }
    }

    const char* decode_type_str = get_decode_type_str(decode_type);
    printf("Using %s decoders\n", decode_type_str);

    // Other arguments
    if (args.is_show_list) {
        SELECT_DECODE_TYPE(decode_type, {
            using factory_t = it1;
            list_codes<factory_t>();
        });
        return tl::unexpected(0);
    }

    if ((args.code_id < 0) || (args.code_id >= COMMON_CODES.N)) {
        fprintf(
            stderr, 
            "Config must be between %d...%d\n"
            "Run '%s -l' for list of codes\n", 
            0, int(COMMON_CODES.N-1), argv[0]);
        return tl::unexpected(1);
    }

    if (args.noise_level < 0) {
        fprintf(stderr, "Noise level must be positive\n");
        return tl::unexpected(1);
    }

    // NOTE: Hard decision decoding has hard upper limit on noise level
    if (decode_type == DecodeType::HARD8) {
        if ((args.noise_level < 0) || (args.noise_level > NOISE_MAX)) {
            fprintf(
                stderr,
                "Hard decision noise level must be between %d...%d\n",
                0, NOISE_MAX
            );
            return tl::unexpected(1);
        }
    }

    if (args.total_input_bytes < 0) {
        fprintf(stderr, "Total input bytes must be positive\n");
        return tl::unexpected(1);
    }

    auto simd_type = SIMD_Type_List.back();
    if (args.simd_type_str != NULL) {
        if (strncmp(args.simd_type_str, "scalar", 7) == 0) {
            simd_type = SIMD_Type::SCALAR;
        #if defined(VITERBI_SIMD_X86)
        } else if (strncmp(args.simd_type_str, "sse", 4) == 0) {
            simd_type = SIMD_Type::SIMD_SSE;
        } else if (strncmp(args.simd_type_str, "avx", 4) == 0) {
            simd_type = SIMD_Type::SIMD_AVX;
        #elif defined(VITERBI_SIMD_ARM)
        } else if (strncmp(args.simd_type_str, "neon", 5) == 0) {
            simd_type = SIMD_Type::SIMD_NEON;
        #endif
        } else {
            fprintf(
                stderr, 
                "Invalid option for decode_type='%s'\n"
                "Run '%s -h' for description of '-d'\n", 
                args.simd_type_str, 
                argv[0]);
            return tl::unexpected(1);
        }
    }

    // Generate seed
    if (args.is_randomise_seed) {
        const auto dt_now = std::chrono::system_clock::now().time_since_epoch();
        const auto us_now = std::chrono::duration_cast<std::chrono::microseconds>(dt_now).count();
        std::srand((unsigned int)us_now);
        args.random_seed = std::rand();
        printf("Using random_seed=%d\n", args.random_seed);
    }
    std::srand((unsigned int)args.random_seed);

    bool is_soft_noise = true;
    switch (decode_type) {
    case DecodeType::SOFT16: is_soft_noise = true; break;
    case DecodeType::SOFT8: is_soft_noise = true; break;
    case DecodeType::HARD8: is_soft_noise = false; break;
    default: break;
    }

    Arguments out;
    out.code_id = size_t(args.code_id);
    out.is_soft_noise = is_soft_noise;
    out.noise_level = uint64_t(args.noise_level);
    out.simd_type = simd_type;
    out.decode_type = decode_type;
    out.total_input_bytes = size_t(args.total_input_bytes);
    return out;
}

int main(int argc, char** argv) {
    auto res = parse_args(argc, argv);
    if (!res) {
        return res.error();
    }

    auto& args = res.value();
    SELECT_COMMON_CODES(args.code_id, {
        const auto& code = it;
        select_code(code, args);
    });
    return 0;
}

template <size_t K, size_t R, typename code_t>
void select_code(const Code<K,R,code_t>& code, Arguments args) {
    SELECT_DECODE_TYPE(args.decode_type, {
        auto config = it0;
        using factory_t = it1;
        init_test<factory_t>(code, config, args);
    });
}

template <class factory_t, size_t K, size_t R, typename code_t, typename soft_t, typename error_t>
void init_test(
    const Code<K,R,code_t>& code, 
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    Arguments args
) {
    printf("Using '%s': K=%zu, R=%zu\n", code.name, code.K, code.R);

    const Decoder_Config<soft_t, error_t> config = config_factory(code.R);
    auto enc = ConvolutionalEncoder_ShiftRegister(code.K, code.R, code.G.data());
    auto branch_table = ViterbiBranchTable<K,R,soft_t>(code.G.data(), config.soft_decision_high, config.soft_decision_low);

    // Run decoder
    for (const auto& simd_type: SIMD_Type_List) {
        if (args.simd_type >= simd_type) {
            SELECT_FACTORY_ITEM(factory_t, simd_type, K, R, {
                using decoder_t = it;
                const char* name = get_simd_type_string(simd_type);
                if constexpr(decoder_t::is_valid) {
                    printf("Using %s decoder\n", name);
                    auto vitdec = decoder_t(branch_table, config.decoder_config);
                    run_test(
                        vitdec, &enc, 
                        args.noise_level, args.is_soft_noise, args.total_input_bytes,
                        config.soft_decision_high, config.soft_decision_low
                    );
                } else {
                    printf("Requested %s decoder unsuccessfully\n", name);
                }
            });
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
