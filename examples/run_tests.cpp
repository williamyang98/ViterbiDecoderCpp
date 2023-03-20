#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <inttypes.h>
#include <vector>
#include <random>

#include "convolutional_encoder.h"
#include "convolutional_encoder_shift_register.h"

#include "helpers/common_codes.h"
#include "helpers/decoder_configs.h"
#include "helpers/decoder_factories.h"
#include "helpers/test_helpers.h"
#include "utility/console_colours.h"
#include "getopt/getopt.h"

constexpr int NOISE_MAX = 100;
enum DecodeType {
    SOFT16, SOFT8, HARD8
};

struct GlobalTestResults {
    size_t total_pass = 0;
    size_t total_tests = 0;
};

struct TestResult {
    uint64_t error_metric;
    size_t total_bit_errors;
    size_t total_bits;
};

template <template <size_t, size_t> class factory_t, typename ... U>
void run_tests_on_codes(U&& ... args);

template <template <size_t, size_t> class factory_t, size_t K, size_t R, typename soft_t, typename error_t>
void run_tests(
    const Code<K,R>& code, 
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    GlobalTestResults& global_results,
    const DecodeType decode_type,
    const uint64_t noise_level, const bool is_soft_noise,
    const size_t total_input_bytes 
);

template <typename soft_t, class T>
TestResult run_test(
    T& vitdec, 
    ConvolutionalEncoder* enc, 
    const uint64_t noise_level,
    const bool is_soft_noise,
    const size_t total_input_bytes,
    const soft_t soft_decision_high,
    const soft_t soft_decision_low
);

template <size_t K, size_t R>
void print_test_result(
    TestResult result, 
    const Code<K,R>& code, 
    const DecodeType decode_type,
    const SIMD_Type simd_type
);

void usage() {
    fprintf(stderr, 
        "run_tests, Runs all tests\n\n"
        "    [-h Show usage]\n"
    );
}

int main(int argc, char** argv) {
    const size_t N_max = common_codes.N;
    assert(N_max > 0u);

	int opt; 
    while ((opt = getopt_custom(argc, argv, "h")) != -1) {
        switch (opt) {
        case 'h':
        default:
            usage();
            return 1;
        }
    }

    std::srand(0);
    const uint64_t noise_level = 0;
    const size_t total_input_bytes = 64;
    GlobalTestResults global_results;

    printf(
        "Status | %*s | %*s | %*s |  K  R | Coefficients\n",
        8, "Decoder",
        9, "SIMD",
        16, "Name"
    );
    run_tests_on_codes<ViterbiDecoder_Factory_u16>(get_soft16_decoding_config, global_results, DecodeType::SOFT16, noise_level, true,  total_input_bytes);
    run_tests_on_codes<ViterbiDecoder_Factory_u8> (get_soft8_decoding_config,  global_results, DecodeType::SOFT8,  noise_level, true,  total_input_bytes);
    run_tests_on_codes<ViterbiDecoder_Factory_u8> (get_hard8_decoding_config,  global_results, DecodeType::HARD8,  noise_level, false, total_input_bytes);

    const bool is_pass = (global_results.total_pass == global_results.total_tests);
    printf("\n\n");
    if (is_pass) {
        printf(CONSOLE_GREEN);
    } else {
        printf(CONSOLE_RED);
    }
    printf("PASSED %zu/%zu TESTS\n", global_results.total_pass, global_results.total_tests);
    printf(CONSOLE_RESET);

    return is_pass ? 0 : 1;
}

template <template <size_t, size_t> class factory_t, typename ... U>
void run_tests_on_codes(U&& ... args) {
    run_tests<factory_t>(common_codes.code_0, std::forward<U>(args)...);
    run_tests<factory_t>(common_codes.code_1, std::forward<U>(args)...);
    run_tests<factory_t>(common_codes.code_2, std::forward<U>(args)...);
    run_tests<factory_t>(common_codes.code_3, std::forward<U>(args)...);
    run_tests<factory_t>(common_codes.code_4, std::forward<U>(args)...);
    run_tests<factory_t>(common_codes.code_5, std::forward<U>(args)...);
    run_tests<factory_t>(common_codes.code_6, std::forward<U>(args)...);
    run_tests<factory_t>(common_codes.code_7, std::forward<U>(args)...);
}

template <template <size_t, size_t> class factory_t, size_t K, size_t R, typename soft_t, typename error_t>
void run_tests(
    const Code<K,R>& code, 
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    GlobalTestResults& global_results,
    const DecodeType decode_type,
    const uint64_t noise_level, const bool is_soft_noise,
    const size_t total_input_bytes 
) {
    const Decoder_Config<soft_t, error_t> config = config_factory(code.R);
    auto enc = ConvolutionalEncoder_ShiftRegister(code.K, code.R, code.G.data());
    auto branch_table = ViterbiBranchTable<K,R,soft_t>(code.G.data(), config.soft_decision_high, config.soft_decision_low);

    if constexpr(factory_t<K,R>::Scalar::is_valid) {
        // NOTE: Known issue with SOFT8 scalar decoder where the small range of uint8 error metric results in overflow
        //       This doesn't occur with the SIMD equivalents since they use saturated arithmetic to avoid overflows
        if ((R == 6) && (decode_type == DecodeType::SOFT8)) {
            goto skip_scalar;
        }

        auto vitdec = typename factory_t<K,R>::Scalar(branch_table, config.decoder_config);
        const auto res = run_test(
            vitdec, &enc, 
            noise_level, is_soft_noise, 
            total_input_bytes, 
            config.soft_decision_high, config.soft_decision_low
        );
        print_test_result(res, code, decode_type, SIMD_Type::SCALAR);
        global_results.total_tests++;
        if (res.total_bit_errors == 0) global_results.total_pass++;
    }
    skip_scalar:

    #if defined(VITERBI_SIMD_X86)
    if constexpr(factory_t<K,R>::SIMD_SSE::is_valid) {
        auto vitdec = typename factory_t<K,R>::SIMD_SSE(branch_table, config.decoder_config);
        const auto res = run_test(
            vitdec, &enc, 
            noise_level, is_soft_noise, 
            total_input_bytes, 
            config.soft_decision_high, config.soft_decision_low
        );
        print_test_result(res, code, decode_type, SIMD_Type::SIMD_SSE);
        global_results.total_tests++;
        if (res.total_bit_errors == 0) global_results.total_pass++;
    }

    if constexpr(factory_t<K,R>::SIMD_AVX::is_valid) {
        auto vitdec = typename factory_t<K,R>::SIMD_AVX(branch_table, config.decoder_config);
        const auto res = run_test(
            vitdec, &enc, 
            noise_level, is_soft_noise, 
            total_input_bytes, 
            config.soft_decision_high, config.soft_decision_low
        );
        print_test_result(res, code, decode_type, SIMD_Type::SIMD_AVX);
        global_results.total_tests++;
        if (res.total_bit_errors == 0) global_results.total_pass++;
    } 
    #elif defined(VITERBI_SIMD_ARM)
    if constexpr(factory_t<K,R>::SIMD_NEON::is_valid) {
        auto vitdec = typename factory_t<K,R>::SIMD_NEON(branch_table, config.decoder_config);
        const auto res = run_test(
            vitdec, &enc, 
            noise_level, is_soft_noise, 
            total_input_bytes, 
            config.soft_decision_high, config.soft_decision_low
        );
        print_test_result(res, code, decode_type, SIMD_Type::SIMD_NEON);
        global_results.total_tests++;
        if (res.total_bit_errors == 0) global_results.total_pass++;
    } 
    #endif

    return;
}

template <typename soft_t, class T>
TestResult run_test(
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

    const size_t total_errors = get_total_bit_errors(tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes);

    TestResult res;
    res.error_metric = error;
    res.total_bit_errors = total_errors;
    res.total_bits = total_input_bits;
    return res;
}

const char* get_decode_type_string(DecodeType mode) {
    switch (mode) {
    case DecodeType::SOFT16:  return "SOFT16";
    case DecodeType::SOFT8:   return "SOFT8";
    case DecodeType::HARD8:   return "HARD8";
    default:                    return "UNKNOWN";
    }
}

template <size_t K, size_t R>
void print_test_result(
    TestResult result, 
    const Code<K,R>& code, 
    const DecodeType decode_type,
    const SIMD_Type simd_type
) {
    constexpr bool is_print_colors = true;

    const bool is_failed = (result.total_bit_errors != 0);
    if (is_failed) {
        if (is_print_colors) printf(CONSOLE_RED);
        printf("\n");
        printf("FAILED | ");
    } else {
        if (is_print_colors) printf(CONSOLE_GREEN);
        printf("PASSED | ");
    }

    // Decode type
    printf("%*s | ", 8, get_decode_type_string(decode_type));

    // SIMD type
    printf("%*s | ", 9, get_simd_type_string(simd_type));

    // Code description
    printf("%*s | %2zu %2zu | ", 16, code.name, code.K, code.R);
    // Coefficients in decimal form
    const auto& G = code.G;
    const size_t N = G.size();
    printf("[");
    for (size_t i = 0; i < N; i++) {
        printf("%u", G[i]);
        if (i != (N-1)) {
            printf(",");
        }
    }
    printf("]");

    if (is_failed) {
        // Error message
        printf("\n");
        printf(
            "       | errors=%zu/%zu, error_metric=%" PRIu64 "\n", 
            result.total_bit_errors, result.total_bits, result.error_metric
        );
    }

    printf("\n");
    if (is_print_colors) printf(CONSOLE_RESET);
}