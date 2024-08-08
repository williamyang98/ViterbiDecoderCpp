#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <string>
#include <inttypes.h>
#include <cctype>
#include <vector>
#include <random>
#include <optional>
#include <mutex>

#include "viterbi/convolutional_encoder.h"
#include "viterbi/convolutional_encoder_lookup.h"
#include "viterbi/convolutional_encoder_shift_register.h"
#include "viterbi/viterbi_decoder_core.h"
#include "viterbi/viterbi_branch_table.h"

#include "helpers/common_codes.h"
#include "helpers/simd_type.h"
#include "helpers/decode_type.h"
#include "helpers/test_helpers.h"
#include "helpers/cli_filters.h"
#include "getopt/getopt.h"
#include "utility/timer.h"
#include "utility/span.h"
#include "utility/thread_pool.h"

struct TestResult {
    uint64_t update_symbols_ns;
    uint64_t chainback_bits_ns;
};

struct Arguments {
    float total_duration_seconds;
    size_t total_input_bytes;
    CLI_Filters filters;
};

template <size_t K, size_t R, typename code_t>
void select_code(const Code<K,R,code_t>& code, size_t code_id, Arguments args);

template <class factory_t, size_t K, size_t R, typename code_t, typename soft_t, typename error_t>
void init_test(
    const Code<K,R,code_t>& code, 
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    DecodeType decode_type,
    Arguments args
);

template <class decoder_t, size_t K, size_t R, typename soft_t, typename error_t>
void run_test(
    ViterbiDecoder_Core<K,R,error_t,soft_t>& vitdec, 
    const ViterbiBranchTable<K,R,soft_t>& branch_table,
    const soft_t* symbols, const size_t total_symbols, 
    const uint8_t* in_bytes, uint8_t* out_bytes, const size_t total_input_bytes,
    const float total_duration_seconds,
    std::vector<TestResult>& out_results
);

template <size_t K, size_t R, typename code_t>
void fprintf_results(
    FILE* fp_out, 
    const Code<K,R,code_t>& code, DecodeType decode_type, SIMD_Type simd_type,
    tcb::span<const TestResult> results,
    size_t total_input_bytes, size_t total_symbols
);


void usage() {
    fprintf(stderr, 
        " run_benchmark, Runs benchmark on viterbi decoding\n\n"
        "    [-t <total_threads> (default: 1)]\n"
        "    [-T <total_duration_of_benchmark_seconds> (default: 1.0)]\n"
        "    [-M <total_input_bytes> (default: 256)]\n"
    );
    cli_filters_print_usage();
    fprintf(stderr,
        "    [-h Show usage]\n"
    );
}

static bool g_is_first_result = true;
static std::unique_ptr<ThreadPool> thread_pool = nullptr;
static std::mutex mutex_stderr;
static std::mutex mutex_fp_out;
static FILE* fp_out = stdout;
static std::vector<std::vector<TestResult>> g_per_thread_test_results;

int main(int argc, char** argv) {
    int total_threads = 1;
    float total_duration_seconds = 1.0;
    int total_input_bytes = 256;
    CLI_Filters filters;
    while (true) {
        const int opt = getopt_custom(argc, argv, "t:T:h" CLI_FILTERS_GETOPT_STRING);
        if (opt == -1) break;
        switch (opt) {
            case 't':
                total_threads = atoi(optarg);
                break;
            case 'T':
                total_duration_seconds = float(atof(optarg));
                break;
            case 'M':
                total_input_bytes = atoi(optarg);
                break;
            case 'h':
                usage();
                return 0;
            default: {
                using R = CLI_Filters_Getopt_Result;
                const auto res = cli_filters_parse_getopt(filters, opt, optarg, argv[0]);
                if (res == R::ERROR_PARSE) return 1;
                if (res == R::SUCCESS_EXIT) return 0;
                if (res == R::NONE) {
                    usage();
                    return 1;
                }
                break;
            }
        }
    }

    if (total_threads < 0) {
        fprintf(stderr, "Total threads must be >= 0, got %d\n", total_threads);
        return 1;
    }

    if (total_duration_seconds <= 0.0f) {
        fprintf(stderr, "Duration of benchmark in seconds must be positive (%.3f)\n", total_duration_seconds);
        return 1;
    }

    if (total_input_bytes <= 0) {
        fprintf(stderr, "Total input bytes must be > 0, got %d\n", total_input_bytes);
        return 1;
    }

    Arguments args;
    args.total_duration_seconds = total_duration_seconds;
    args.total_input_bytes = size_t(total_input_bytes);
    args.filters = filters;

    thread_pool = std::make_unique<ThreadPool>(size_t(total_threads));
    g_per_thread_test_results.resize(thread_pool->get_total_threads());
    for (auto& pool: g_per_thread_test_results) {
        pool.reserve(4096);
    }
 
    size_t code_id = 0;
    FOR_COMMON_CODES({
        const auto& code = it;
        select_code(code, code_id, args);
        code_id++;
    });

    const int total_tasks = thread_pool->get_total_tasks();
    fprintf(stderr, "Using %zu threads\n", thread_pool->get_total_threads());
    fprintf(stderr, "Total tasks in thread pool: %d\n", total_tasks);
    if (total_tasks > 0) {
        fprintf(fp_out, "[\n");
        thread_pool->wait_all();
        fprintf(fp_out, "]\n");
    }
    return 0;
}

template <size_t K, size_t R, typename code_t>
void select_code(const Code<K,R,code_t>& code, size_t code_id, Arguments args) {
    if (!args.filters.allow_code_index(code_id)) return;
    for (const auto decode_type: Decode_Type_List) {
        if (!args.filters.allow_decode_type(decode_type)) continue;
        SELECT_DECODE_TYPE(decode_type, {
            auto config = it0;
            using factory_t = it1;
            init_test<factory_t>(code, config, decode_type, args);
        });
    }
}

template <class factory_t, size_t K, size_t R, typename code_t, typename soft_t, typename error_t>
void init_test(
    const Code<K,R,code_t>& code, 
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    DecodeType decode_type, Arguments args
) {
    const Decoder_Config<soft_t, error_t> config = config_factory(code.R);
    for (const auto simd_type: SIMD_Type_List) {
        if (!args.filters.allow_simd_type(simd_type)) continue;
        SELECT_FACTORY_ITEM(factory_t, simd_type, K, R, {
            using decoder_t = it;
            if constexpr(decoder_t::is_valid) {
                thread_pool->push_task([code, config, decode_type, simd_type, args](size_t thread_id) {
                    const float total_duration_seconds = args.total_duration_seconds;
                    auto enc = ConvolutionalEncoder_ShiftRegister(code.K, code.R, code.G.data());
                    auto branch_table = ViterbiBranchTable<K,R,soft_t>(code.G.data(), config.soft_decision_high, config.soft_decision_low);
                    auto vitdec = ViterbiDecoder_Core<K,R,error_t,soft_t>(config.decoder_config);
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
 
                    std::srand(static_cast<unsigned int>(time(NULL)));
                    generate_random_bytes(tx_input_bytes.data(), tx_input_bytes.size());
                    encode_data(
                        &enc, 
                        tx_input_bytes.data(), tx_input_bytes.size(), 
                        output_symbols.data(), output_symbols.size(),
                        config.soft_decision_high, config.soft_decision_low
                    );
 
                    auto& results = g_per_thread_test_results[thread_id];
                    results.clear();

                    vitdec.set_traceback_length(total_input_bits);
                    run_test<decoder_t>(
                        vitdec, branch_table,
                        output_symbols.data(), output_symbols.size(), 
                        tx_input_bytes.data(), rx_input_bytes.data(), total_input_bytes,
                        total_duration_seconds,
                        results
                    );
                    const size_t total_results = results.size();
                    auto lock_stderr = std::scoped_lock(mutex_stderr);
                    fprintf(stderr, "thread=%zu,name='%s',K=%zu,R=%zu,decode=%s,simd=%s,input_bytes=%zu,total_results=%zu\n", 
                        thread_id,
                        code.name, code.K, code.R, 
                        get_decode_type_str(decode_type), get_simd_type_string(simd_type), 
                        total_input_bytes, total_results
                    );
                    auto lock_fp_out = std::scoped_lock(mutex_fp_out);
                    fprintf_results(fp_out, code, decode_type, simd_type, results, total_input_bytes, output_symbols.size());
                });
            }
        });
    }
}

template <class decoder_t, size_t K, size_t R, typename soft_t, typename error_t>
void run_test(
    ViterbiDecoder_Core<K,R,error_t,soft_t>& vitdec, 
    const ViterbiBranchTable<K,R,soft_t>& branch_table,
    const soft_t* symbols, const size_t total_symbols, 
    const uint8_t* in_bytes, uint8_t* out_bytes, const size_t total_input_bytes,
    const float total_duration_seconds,
    std::vector<TestResult>& out_results
) {
    const size_t total_input_bits = total_input_bytes*8u;

    Timer total_time;
    size_t curr_iteration = 0;
    while (true) {
        const float seconds_elapsed = float(total_time.get_delta<std::chrono::milliseconds>())*1e-3f;
        if (seconds_elapsed > total_duration_seconds) {
            break;
        }
        TestResult result;
        {
            Timer t;
            vitdec.reset();
        }
        {
            Timer t;
            const uint64_t accumulated_error = decoder_t::template update<uint64_t>(vitdec, symbols, total_symbols, branch_table);
            result.update_symbols_ns = t.get_delta();
        }
        {
            Timer t;
            vitdec.chainback(out_bytes, total_input_bits, 0u);
            result.chainback_bits_ns = t.get_delta();
        }
        out_results.push_back(result);
    }
}

template <typename T, typename F>
void fprintf_list(FILE* fp_out, const char* formatter, tcb::span<const T> list, F&& func) {
    fprintf(fp_out, "[");
    const size_t N = list.size();
    for (size_t i = 0; i < N; i++) {
        fprintf(fp_out, formatter, func(list[i]));
        if (i < (N-1)) printf(",");
    }
    fprintf(fp_out, "]");
}

template <size_t K, size_t R, typename code_t>
void fprintf_results(
    FILE* fp_out, 
    const Code<K,R,code_t>& code, DecodeType decode_type, SIMD_Type simd_type,
    tcb::span<const TestResult> results,
    size_t total_input_bytes, size_t total_symbols
) {
    if (!g_is_first_result) {
        fprintf(fp_out, ",\n");
    } else {
        g_is_first_result = false;
    }
    fprintf(fp_out, "{\n");
    fprintf(fp_out, " \"name\": \"%s\",\n", code.name);
    fprintf(fp_out, " \"decode_type\": \"%s\",\n", get_decode_type_str(decode_type));
    fprintf(fp_out, " \"simd_type\": \"%s\",\n", get_simd_type_string(simd_type));
    fprintf(fp_out, " \"K\": %zu,\n", code.K);
    fprintf(fp_out, " \"R\": %zu,\n", code.R);
    fprintf(fp_out, " \"G\": ");
    fprintf_list(fp_out, "%u", tcb::span<const code_t>(code.G), [](const auto& e) { return e; });
    fprintf(fp_out, ",\n");
    fprintf(fp_out, " \"total_input_bits\": %zu,\n", total_input_bytes*8);
    fprintf(fp_out, " \"total_symbols\": %zu,\n", total_symbols);
    fprintf(fp_out, " \"update_symbols_ns\": ");
    fprintf_list(fp_out, "%" PRIu64, tcb::span<const TestResult>(results), [](const auto& e) { return e.update_symbols_ns; });
    fprintf(fp_out, ",\n");
    fprintf(fp_out, " \"chainback_bits_ns\": ");
    fprintf_list(fp_out, "%" PRIu64, tcb::span<const TestResult>(results), [](const auto& e) { return e.chainback_bits_ns; });
    fprintf(fp_out, "\n");
    fprintf(fp_out, "}");
}