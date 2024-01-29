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
#include "viterbi/viterbi_decoder_core.h"

#include "helpers/common_codes.h"
#include "helpers/simd_type.h"
#include "helpers/decode_type.h"
#include "helpers/test_helpers.h"
#include "utility/timer.h"
#include "utility/expected.hpp"
#include "getopt/getopt.h"

struct TestResults {
    float update_symbols_per_ms = 0.0f;
    float reset_bits_per_ms = 0.0f;
    float chainback_bits_per_ms = 0.0f;
    float bit_error_rate = 0.0f;      
    size_t total_incorrect_bits = 0u;
    size_t total_decoded_bits = 0u;
    size_t total_error = 0u;
    size_t total_runs = 0u;
};

struct Arguments {
    size_t code_id;
    DecodeType decode_type;
    size_t total_input_bytes;
    float total_duration_seconds;
};

template <size_t K, size_t R, typename code_t>
void select_code(const Code<K,R,code_t>& code, Arguments args);

template <class factory_t, size_t K, size_t R, typename code_t, typename soft_t, typename error_t>
void init_test(
    const Code<K,R,code_t>& code, 
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    Arguments args
);

template <class decoder_t, size_t K, size_t R, typename soft_t, typename error_t>
TestResults run_test(
    ViterbiDecoder_Core<K,R,error_t,soft_t>& vitdec, 
    const soft_t* symbols, const size_t total_symbols, 
    const uint8_t* in_bytes, uint8_t* out_bytes, const size_t total_input_bytes,
    const float total_duration_seconds
);

void usage() {
    fprintf(stderr, 
        " run_benchmark, Runs benchmark on viterbi decoding\n\n"
        "    [-c <code id> (default: 0)]\n"
        "    [-d <decode_type> (default: soft_16)]\n"
        "        soft_16: use u16 error type and soft decision boundaries\n"
        "        soft_8:  use u8  error type and soft decision boundaries\n"
        "        hard_8:  use u8  error type and hard decision boundaries\n"
        "    [-L <total input bytes> (default: 1024)]\n"
        "    [-T <total duration of benchmark> (default: 1.0) ]\n"
        "    [-l Lists all available codes]\n"
        "    [-h Show usage]\n"
    );
}

tl::expected<Arguments, int> parse_args(int argc, char** argv) {
    struct {
        int code_id = 0;
        int total_input_bytes = 1024;
        float total_duration_seconds = 1.0;
        bool is_show_list = false;
        const char* decode_type_str = NULL;
    } args;

	int opt; 
    while ((opt = getopt_custom(argc, argv, "c:d:L:T:lh")) != -1) {
        switch (opt) {
        case 'c':
            args.code_id = atoi(optarg);
            break;
        case 'd':
            args.decode_type_str = optarg;
            break;
        case 'L':
            args.total_input_bytes = atoi(optarg);
            break;
        case 'l':
            args.is_show_list = true;
            break;
        case 'T':
            args.total_duration_seconds = float(atof(optarg));
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

    if (args.total_duration_seconds <= 0.0f) {
        fprintf(stderr, "Duration of benchmark in seconds must be positive (%.3f)\n", args.total_duration_seconds);
        return tl::unexpected(1);
    }

    if (args.total_input_bytes < 0) {
        fprintf(stderr, "Total input bytes must be positive\n");
        return tl::unexpected(1);
    }

    Arguments out;
    out.code_id = size_t(args.code_id);
    out.decode_type = decode_type;
    out.total_input_bytes = size_t(args.total_input_bytes);
    out.total_duration_seconds = args.total_duration_seconds;
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
    auto vitdec = ViterbiDecoder_Core<K,R,error_t,soft_t>(branch_table, config.decoder_config);

    // Generate test data
    const size_t total_input_bytes = args.total_input_bytes;
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

    // compare results
    auto print_independent = [](const TestResults src) {
        printf("update        = %.3f symbols/ms\n", src.update_symbols_per_ms);
        printf("chainback     = %.3f bits/ms\n", src.chainback_bits_per_ms);
        printf("reset         = %.3f bits/ms\n", src.reset_bits_per_ms);
        printf("ber           = %.4f\n", src.bit_error_rate);
        printf("errors        = %zu/%zu\n", src.total_incorrect_bits, src.total_decoded_bits);
        printf("error_metric  = %" PRIu64 "\n",  src.total_error);
    };

    auto print_comparison = [](const TestResults ref, const TestResults src) {
        const float update_speedup = src.update_symbols_per_ms / ref.update_symbols_per_ms;
        const float chainback_speedup = src.chainback_bits_per_ms / ref.chainback_bits_per_ms;
        const float reset_speedup = src.reset_bits_per_ms / ref.reset_bits_per_ms;
        printf("update        = %.3f symbols/ms (x%.2f)\n", src.update_symbols_per_ms, update_speedup);
        printf("chainback     = %.3f bits/ms (x%.2f)\n", src.chainback_bits_per_ms, chainback_speedup);
        printf("reset         = %.3f bits/ms (x%.2f)\n", src.reset_bits_per_ms, reset_speedup);
        printf("ber           = %.4f\n", src.bit_error_rate);
        printf("errors        = %zu/%zu\n", src.total_incorrect_bits, src.total_decoded_bits);
        printf("error_metric  = %" PRIu64 "\n",  src.total_error);
    };

    // Run tests
    const float total_duration_seconds = args.total_duration_seconds;
    TestResults scalar_test_results;
    for (const auto& simd_type: SIMD_Type_List) {
        SELECT_FACTORY_ITEM(factory_t, simd_type, K, R, {
            using decoder_t = it;
            if constexpr(decoder_t::is_valid) {
                vitdec.set_traceback_length(total_input_bits);
                const auto res = run_test<decoder_t>(
                    vitdec, 
                    output_symbols.data(), output_symbols.size(), 
                    tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes,
                    total_duration_seconds
                );

                printf("> %s results\n", get_simd_type_string(simd_type));
                if (simd_type == SIMD_Type::SCALAR) {
                    print_independent(res);
                    scalar_test_results = res;
                } else {
                    print_comparison(scalar_test_results, res);
                }
                printf("\n");
            }
        });
    }
}

template <class decoder_t, size_t K, size_t R, typename soft_t, typename error_t>
TestResults run_test(
    ViterbiDecoder_Core<K,R,error_t,soft_t>& vitdec, 
    const soft_t* symbols, const size_t total_symbols, 
    const uint8_t* in_bytes, uint8_t* out_bytes, const size_t total_input_bytes,
    const float total_duration_seconds
) {
    const size_t total_input_bits = total_input_bytes*8u;
    constexpr size_t print_rate = 1u;


    TestResults results;
    uint64_t update_ns = 0;
    uint64_t update_total_symbols = 0;
    uint64_t reset_ns = 0;
    uint64_t reset_total_bits = 0;
    uint64_t chainback_ns = 0;
    uint64_t chainback_total_bits = 0;

    Timer total_time;
    size_t curr_iteration = 0;
    while (true) {
        const float seconds_elapsed = float(total_time.get_delta<std::chrono::milliseconds>())*1e-3f;
        if (seconds_elapsed > total_duration_seconds) {
            break;
        }
        curr_iteration++;
        if (curr_iteration % 10 == 0) {
            printf("Run: %.2f/%.2f\r", seconds_elapsed, total_duration_seconds);
        }

        {
            Timer t;
            vitdec.reset();
            reset_ns += t.get_delta();
            reset_total_bits += vitdec.get_traceback_length();
        }
        {
            Timer t;
            const uint64_t accumulated_error = decoder_t::template update<uint64_t>(vitdec, symbols, total_symbols);
            update_ns += t.get_delta();
            update_total_symbols += total_symbols;

            const error_t normalised_error = vitdec.get_error();
            results.total_error += accumulated_error + uint64_t(normalised_error);
        }
        {
            Timer t;
            vitdec.chainback(out_bytes, total_input_bits, 0u);
            chainback_ns += t.get_delta();
            chainback_total_bits += total_input_bits;
        }
        {
        }
        const size_t total_bit_errors = get_total_bit_errors(in_bytes, out_bytes, total_input_bytes);
        results.total_incorrect_bits += total_bit_errors;
        results.total_decoded_bits += total_input_bits;
        results.total_runs++;
    }
    printf("%*s\r", 100, "");

    results.bit_error_rate = float(results.total_incorrect_bits) / float(results.total_decoded_bits);

    const float rescale_ns_to_ms = 1e+6f;
    results.update_symbols_per_ms = float(update_total_symbols) / float(update_ns)    * rescale_ns_to_ms;
    results.reset_bits_per_ms     = float(reset_total_bits)     / float(reset_ns)     * rescale_ns_to_ms;
    results.chainback_bits_per_ms = float(chainback_total_bits) / float(chainback_ns) * rescale_ns_to_ms;
    return results;
}