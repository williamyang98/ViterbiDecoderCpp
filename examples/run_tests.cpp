#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <inttypes.h>
#include <vector>
#include <map>
#include <random>

#include "viterbi/convolutional_encoder.h"
#include "viterbi/convolutional_encoder_shift_register.h"
#include "viterbi/viterbi_decoder_core.h"

#include "helpers/common_codes.h"
#include "helpers/simd_type.h"
#include "helpers/decode_type.h"
#include "helpers/test_helpers.h"
#include "utility/console_colours.h"
#include "getopt/getopt.h"

struct GlobalTestResults {
    size_t total_pass = 0;
    size_t total_tests = 0;
    size_t total_skipped = 0;

    bool is_pass() const {
        return total_pass == total_tests;
    }
};

struct TestResult {
    uint64_t error_metric;
    size_t total_bit_errors;
    size_t total_bits;
};

class TestKey 
{
private:
    SIMD_Type simd_type;
    DecodeType decode_type;
    size_t K;
    size_t R;
public:
    TestKey(SIMD_Type _simd_type, DecodeType _decode_type, size_t _K, size_t _R)
    : simd_type(_simd_type), decode_type(_decode_type), K(_K), R(_R) {}

    bool operator<(const TestKey& other) const {
        return get_value() < other.get_value();
    }

    uint64_t get_value() const {
        uint64_t value = 0;
        value |= uint64_t(simd_type) << 0;
        value |= uint64_t(decode_type) << 16;
        value |= uint64_t(K) << 32;
        value |= uint64_t(R) << 48;
        return value;
    }
};

std::map<TestKey, const char*> SKIP_TESTS = {
    { TestKey(SIMD_Type::SCALAR, DecodeType::SOFT8, 15, 6), "Overflow in metrics due to high code rate and non saturating arithmetic" },
};

template <class factory_t, typename ... U>
void select_codes(U&& ... args);

template <class factory_t, size_t K, size_t R, typename code_t, typename soft_t, typename error_t>
void run_tests(
    const Code<K,R,code_t>& code, 
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    GlobalTestResults& global_results,
    const DecodeType decode_type,
    const size_t total_input_bytes 
);

template <class decoder_t, size_t K, size_t R, typename soft_t, typename error_t>
TestResult run_test(
    ViterbiDecoder_Core<K,R,error_t,soft_t>& vitdec, 
    ConvolutionalEncoder* enc, 
    const size_t total_input_bytes,
    const soft_t soft_decision_high,
    const soft_t soft_decision_low
);

void print_header();

template <size_t K, size_t R, typename code_t>
void print_code(const Code<K,R,code_t>& code);

template <size_t K, size_t R, typename code_t>
void print_skip_message(
    const Code<K,R,code_t>& code, 
    const DecodeType decode_type,
    const SIMD_Type simd_type,
    const char* message
);

template <size_t K, size_t R, typename code_t>
void print_test_result(
    TestResult result, 
    const Code<K,R,code_t>& code, 
    const DecodeType decode_type,
    const SIMD_Type simd_type
);

void print_summary(const GlobalTestResults& results);

void usage() {
    fprintf(stderr, 
        "run_tests, Runs all tests\n\n"
        "    [-h Show usage]\n"
    );
}

int main(int argc, char** argv) {
    int opt; 
    while ((opt = getopt_custom(argc, argv, "h")) != -1) {
        switch (opt) {
        case 'h':
        default:
            usage();
            return 1;
        }
    }

    const size_t total_input_bytes = 64;
    GlobalTestResults global_results;

    print_header();
    for (const auto& decode_type: Decode_Type_List) {
        SELECT_DECODE_TYPE(decode_type, {
            auto config = it0;
            using factory_t = it1;
            select_codes<factory_t>(config, global_results, decode_type, total_input_bytes);
        });
    }

    print_summary(global_results);
    return global_results.is_pass() ? 0 : 1;
}

template <class factory_t, typename ... U>
void select_codes(U&& ... args) {
    FOR_COMMON_CODES({
        const auto& code = it;
        run_tests<factory_t>(code, std::forward<U>(args)...);
    });
}

template <class factory_t, size_t K, size_t R, typename code_t, typename soft_t, typename error_t>
void run_tests(
    const Code<K,R,code_t>& code, 
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    GlobalTestResults& global_results,
    const DecodeType decode_type,
    const size_t total_input_bytes 
) {
    const Decoder_Config<soft_t, error_t> config = config_factory(code.R);
    auto enc = ConvolutionalEncoder_ShiftRegister(code.K, code.R, code.G.data());
    auto branch_table = ViterbiBranchTable<K,R,soft_t>(code.G.data(), config.soft_decision_high, config.soft_decision_low);
    auto vitdec = ViterbiDecoder_Core<K,R,error_t,soft_t>(branch_table, config.decoder_config);

    for (const auto& simd_type: SIMD_Type_List) {
        SELECT_FACTORY_ITEM(factory_t, simd_type, K, R, {
            using decoder_t = it;
            if constexpr(decoder_t::is_valid) {
                auto skip_key = TestKey(simd_type, decode_type, K, R);
                const auto& skip_entry = SKIP_TESTS.find(skip_key);
                if (skip_entry != SKIP_TESTS.end()) {
                    const char* reason = skip_entry->second;
                    print_skip_message(code, decode_type, simd_type, reason);
                    global_results.total_skipped++;
                } else {
                    const auto res = run_test<decoder_t>(
                        vitdec, &enc, 
                        total_input_bytes, 
                        config.soft_decision_high, config.soft_decision_low
                    );
                    print_test_result(res, code, decode_type, simd_type);
                    global_results.total_tests++;
                    if (res.total_bit_errors == 0) {
                        global_results.total_pass++;
                    }
                }
            } 
        });
    }
}

template <class decoder_t, size_t K, size_t R, typename soft_t, typename error_t>
TestResult run_test(
    ViterbiDecoder_Core<K,R,error_t,soft_t>& vitdec, 
    ConvolutionalEncoder* enc, 
    const size_t total_input_bytes,
    const soft_t soft_decision_high,
    const soft_t soft_decision_low
) {
    assert(vitdec.K == enc->K);
    assert(vitdec.R == enc->R);
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

    const size_t total_output_symbols = output_symbols.size();
    vitdec.reset();
    const uint64_t accumulated_error = decoder_t::template update<uint64_t>(vitdec, output_symbols.data(), total_output_symbols);
    const uint64_t error = accumulated_error + uint64_t(vitdec.get_error());
    vitdec.chainback(rx_input_bytes.data(), total_input_bits, 0u);

    const size_t total_errors = get_total_bit_errors(tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes);

    TestResult res;
    res.error_metric = error;
    res.total_bit_errors = total_errors;
    res.total_bits = total_input_bits;
    return res;
}

void print_header() {
    printf(
        "Status | %*s | %*s | %*s |  K  R | Coefficients\n",
        8, "Decoder",
        9, "SIMD",
        16, "Name"
    );
}

template <size_t K, size_t R, typename code_t>
void print_code(const Code<K,R,code_t>& code) {
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
}

template <size_t K, size_t R, typename code_t>
void print_skip_message(
    const Code<K,R,code_t>& code, 
    const DecodeType decode_type,
    const SIMD_Type simd_type,
    const char* message
) {
    printf("SKIP   | ");
    printf("%*s | ", 8, get_decode_type_str(decode_type));
    printf("%*s | ", 9, get_simd_type_string(simd_type));
    printf("%*s | %2zu %2zu | ", 16, code.name, code.K, code.R);
    print_code(code);
    printf("\n");
    printf("       | Reason: '%s'\n", message);
}

template <size_t K, size_t R, typename code_t>
void print_test_result(
    TestResult result, 
    const Code<K,R,code_t>& code, 
    const DecodeType decode_type,
    const SIMD_Type simd_type
) {
    constexpr bool is_print_colors = true;

    const bool is_failed = (result.total_bit_errors != 0);
    if (is_failed) {
        if (is_print_colors) printf(CONSOLE_RED);
        printf("FAILED | ");
    } else {
        if (is_print_colors) printf(CONSOLE_GREEN);
        printf("PASSED | ");
    }

    printf("%*s | ", 8, get_decode_type_str(decode_type));
    printf("%*s | ", 9, get_simd_type_string(simd_type));
    printf("%*s | %2zu %2zu | ", 16, code.name, code.K, code.R);
    print_code(code);

    printf("\n");
    if (is_failed) {
        printf(
            "       | Got unexpected errors in output: bit_errors=%zu/%zu, error_metric=%" PRIu64 ".\n", 
            result.total_bit_errors, result.total_bits, result.error_metric
        );
    } 
    if (is_print_colors) printf(CONSOLE_RESET);
}

void print_summary(const GlobalTestResults& results) {
    printf("\n\n");
    if (results.is_pass()) {
        printf(CONSOLE_GREEN);
    } else {
        printf(CONSOLE_RED);
    }
    printf("PASSED %zu/%zu TESTS\n", results.total_pass, results.total_tests);
    printf(CONSOLE_RESET);

    if (results.total_skipped > 0) {
        printf("SKIPPED %zu TESTS\n", results.total_skipped);
    }
}