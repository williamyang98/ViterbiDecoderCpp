#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <inttypes.h>
#include <vector>
#include <map>
#include <random>
#include <optional>

#include "viterbi/convolutional_encoder.h"
#include "viterbi/convolutional_encoder_shift_register.h"
#include "viterbi/viterbi_decoder_core.h"

#include "helpers/common_codes.h"
#include "helpers/simd_type.h"
#include "helpers/decode_type.h"
#include "helpers/test_helpers.h"
#include "helpers/cli_filters.h"
#include "getopt/getopt.h"
#include "utility/span.h"
#include "utility/thread_pool.h"
#include "utility/timer.h"

struct TestRange {
    float EbNo_dB_initial;
    float EbNo_dB_step;
    size_t maximum_generated_bits;
};

struct Arguments {
    size_t maximum_error_bits;
    size_t traceback_length_bytes;
    size_t maximum_data_points;
    uint64_t random_seed;
    float maximum_generated_bits_scale;
    std::optional<float> timeout_seconds;
    CLI_Filters filters;
};

struct TestResults {
    std::vector<float> EbNo_dB;
    std::vector<float> bit_error_rates;
    std::vector<size_t> total_bit_errors;
    std::vector<size_t> total_bits;
};

TestRange get_test_range(const size_t K, const size_t R);

template <size_t K, size_t R, typename code_t>
void select_decode_type(const Code<K,R,code_t>& code, const size_t code_id, const Arguments& args);

template <class factory_t, size_t K, size_t R, typename code_t, typename soft_t, typename error_t>
void run_tests(
    const Code<K,R,code_t>& code, const DecodeType decode_type,
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    const Arguments& args
);

template <class decoder_t, size_t K, size_t R, typename soft_t, typename error_t, typename code_t>
TestResults run_test(
    ViterbiDecoder_Core<K,R,error_t,soft_t>& vitdec, ConvolutionalEncoder* enc, 
    const soft_t soft_decision_high, const soft_t soft_decision_low,
    const Arguments& args, const TestRange& test_range, 
    const Code<K,R,code_t>& code, const DecodeType decode_type, const SIMD_Type simd_type, const size_t thread_id
);

template <size_t K, size_t R, typename code_t>
void print_test_results(
    FILE* fp_out, 
    const Code<K,R,code_t>& code, 
    const DecodeType decode_type, const SIMD_Type simd_type,
    const TestResults& results
);

void usage() {
    fprintf(stderr, 
        "run_tests, Runs all tests\n\n"
        "    [-t <total_threads> (default: 0)]\n"
        "    [-L <traceback_length> (default: 512)]\n"
        "    [-n <maximum_error_bits> (default: 1024)]\n"
        "    [-D <maximum_data_points> (default: 30)]\n"
        "    [-S <random_seed> (default: 0) ]\n"
        "    [-k <maximum_generated_bits_scale> (default: 1.0)]\n"
        "    [-T <timeout_seconds> (default: None)]\n"
    );
    cli_filters_print_usage();
    fprintf(stderr,
        "    [-h Show usage]\n"
    );
}

static std::unique_ptr<ThreadPool> thread_pool = nullptr;
static bool g_is_first_result = true;
static std::mutex mutex_stderr;
static std::mutex mutex_fp_out;
static FILE* fp_out = stdout;

int main(int argc, char** argv) {
    int total_threads = 0;
    int traceback_length = 512;
    int maximum_error_bits = 1024;
    int maximum_data_points = 30;
    float maximum_generated_bits_scale = 1.0f;
    std::optional<float> timeout_seconds = std::nullopt;
    int random_seed = 0;
    CLI_Filters filters;
    while (true) {
        const int opt = getopt_custom(argc, argv, "t:L:n:D:S:k:T:h" CLI_FILTERS_GETOPT_STRING);
        if (opt == -1) break;
        switch (opt) {
            case 't':
                total_threads = atoi(optarg);
                break;
            case 'L':
                traceback_length = atoi(optarg);
                break;
            case 'n':
                maximum_error_bits = atoi(optarg);
                break;
            case 'D':
                maximum_data_points = atoi(optarg);
                break;
            case 'S':
                random_seed = atoi(optarg);
                break;
            case 'k':
                maximum_generated_bits_scale = atof(optarg);
                break;
            case 'T':
                timeout_seconds = std::optional(atof(optarg));
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

    if (traceback_length <= 0) {
        fprintf(stderr, "Maximum number of error bits must be > 0, got %d\n", traceback_length);
        return 1;
    }

    if (maximum_error_bits <= 0) {
        fprintf(stderr, "Maximum number of error bits must be > 0, got %d\n", maximum_error_bits);
        return 1;
    }

    if (maximum_data_points <= 0) {
        fprintf(stderr, "Maximum number of data points must be > 0, got %d\n", maximum_data_points);
        return 1;
    }

    if (random_seed < 0) {
        fprintf(stderr, "Random seed must be >= 0, got %d\n", random_seed);
        return 1;
    }

    if (maximum_generated_bits_scale <= 0.0f) {
        fprintf(stderr, "Maximum generated bits scale must be > 0, got %f\n", maximum_generated_bits_scale);
        return 1;
    }

    if (timeout_seconds.has_value() && timeout_seconds.value() <= 0.0f) {
        fprintf(stderr, "Timeout must be > 0, got %f\n", timeout_seconds.value());
        return 1;
    }
 
    Arguments args;
    args.traceback_length_bytes = size_t(traceback_length);
    args.maximum_error_bits = size_t(maximum_error_bits);
    args.maximum_data_points = size_t(maximum_data_points);
    args.maximum_generated_bits_scale = maximum_generated_bits_scale;
    args.random_seed = 0;
    args.timeout_seconds = timeout_seconds;
    args.filters = filters;
    if (random_seed == 0) {
        args.random_seed = uint64_t(time(NULL));
    } else {
        args.random_seed = uint64_t(random_seed);
    }

    thread_pool = std::make_unique<ThreadPool>(size_t(total_threads));
 
    size_t code_id = 0;
    FOR_COMMON_CODES({
        const auto& code = it;
        select_decode_type(code, code_id, args);
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

TestRange get_test_range(const size_t K, const size_t R) {
    // estimate error correcting capability as average_hamming_distance * code_rate
    // average_hamming_distance = constraint_length/2
    // ecc ∝ K*R
    // runtime ∝ R * 2^(K-1)
    const size_t runtime_scale = R * (1<<(K-1));
    const size_t error_correcting_capability = K*R;
    size_t base_total_bits = size_t(1e9); 
    TestRange range;
    range.EbNo_dB_initial = -std::ceil(std::pow(float(error_correcting_capability), 0.8f));
    // TODO: Figure out a better way to calculate this that generalises
    if (K >= 9) {
        range.EbNo_dB_initial = -17.0f;
        base_total_bits = size_t(1e10);
    }
    // Measure the sharp cutoff of code with high error correction ability
    range.EbNo_dB_step = (error_correcting_capability > 20) ? 0.5f : 1.0f;
    range.maximum_generated_bits = base_total_bits / runtime_scale;
    return range;
}

template <size_t K, size_t R, typename code_t>
void select_decode_type(const Code<K,R,code_t>& code, const size_t code_id, const Arguments& args) {
    if (!args.filters.allow_code_index(code_id)) return;
    for (const auto& decode_type: Decode_Type_List) {
        if (!args.filters.allow_decode_type(decode_type)) continue;
        SELECT_DECODE_TYPE(decode_type, {
            auto config_factory = it0;
            using factory_t = it1;
            run_tests<factory_t>(code, decode_type, config_factory, args);
        });
    }

}

template <class factory_t, size_t K, size_t R, typename code_t, typename soft_t, typename error_t>
void run_tests(
    const Code<K,R,code_t>& code, const DecodeType decode_type,
    Decoder_Config<soft_t,error_t>(*config_factory)(const size_t),
    const Arguments& args
) {
    for (const auto& simd_type: SIMD_Type_List) {
        if (!args.filters.allow_simd_type(simd_type)) continue;
        SELECT_FACTORY_ITEM(factory_t, simd_type, K, R, {
            using decoder_t = it;
            if constexpr(decoder_t::is_valid) {
                thread_pool->push_task([=](size_t thread_id) {
                    const Decoder_Config<soft_t, error_t> config = config_factory(code.R);
                    auto branch_table = ViterbiBranchTable<K,R,soft_t>(code.G.data(), config.soft_decision_high, config.soft_decision_low);
                    auto enc = ConvolutionalEncoder_ShiftRegister(code.K, code.R, code.G.data());
                    auto vitdec = ViterbiDecoder_Core<K,R,error_t,soft_t>(branch_table, config.decoder_config);
                    const auto test_range = get_test_range(K,R);
                    const auto test_results = run_test<decoder_t>(
                        vitdec, &enc, 
                        config.soft_decision_high, config.soft_decision_low,
                        args, test_range, 
                        code, decode_type, simd_type, thread_id
                    ); 
                    auto lock_fp_out = std::scoped_lock(mutex_fp_out);
                    print_test_results(fp_out, code, decode_type, simd_type, test_results);
                });
            } 
        });
    }
}

template <class decoder_t, size_t K, size_t R, typename soft_t, typename error_t, typename code_t>
TestResults run_test(
    ViterbiDecoder_Core<K,R,error_t,soft_t>& vitdec, ConvolutionalEncoder* enc, 
    const soft_t soft_decision_high, const soft_t soft_decision_low,
    const Arguments& args, const TestRange& test_range, 
    const Code<K,R,code_t>& code, const DecodeType decode_type, const SIMD_Type simd_type, const size_t thread_id
) {
    assert(vitdec.K == enc->K);
    assert(vitdec.R == enc->R);
    // determine size of buffers per block
    const size_t total_block_bytes = args.traceback_length_bytes;
    const size_t total_block_bits = total_block_bytes*8u;
    size_t total_block_symbols = 0;
    {
        const size_t total_tail_bits = K-1u;
        const size_t total_data_bits = total_block_bytes*8;
        const size_t total_bits = total_data_bits + total_tail_bits;
        total_block_symbols = total_bits * R;
    }
    // setup buffers to generate data in blocks
    std::vector<uint8_t> tx_block_bytes;
    std::vector<uint8_t> rx_block_bytes;
    std::vector<float> output_symbols_float;
    std::vector<soft_t> output_symbols; 
    std::vector<float> bit_error_rates;
    vitdec.set_traceback_length(total_block_bits);
    tx_block_bytes.resize(total_block_bytes);
    rx_block_bytes.resize(total_block_bytes);
    output_symbols_float.resize(total_block_symbols);
    output_symbols.resize(total_block_symbols);
    // determine offset and scale for normalised symbols
    const float symbol_norm_mean = (float(soft_decision_high) + float(soft_decision_low)) / 2.0f;
    const float symbol_norm_magnitude = (float(soft_decision_high) - float(soft_decision_low)) / 2.0f;
 
    TestResults results;
    const size_t max_generated_bits = size_t(std::ceil(args.maximum_generated_bits_scale*float(test_range.maximum_generated_bits)));
    // Move seed here to avoid infinite loop for high EbNo_dB
    std::mt19937 rand_engine{(unsigned int)(args.random_seed)};
    for (size_t curr_point = 0; ; curr_point++) {
        const float EbNo_dB = test_range.EbNo_dB_initial + float(curr_point)*test_range.EbNo_dB_step;
        const float snr_dB = EbNo_dB + 10.0f*std::log10(float(R));
        // E(X^2) = Var(X) + [E(X)]^2 = Var(X), since E(X) = 0
        const float noise_variance = std::pow(10.0f, -snr_dB/10.0f);
        const float noisy_signal_energy = 1.0f + noise_variance;
        const float noisy_signal_norm = 1.0f/std::sqrt(noisy_signal_energy);
        std::normal_distribution<float> rand_norm_dist(0.0f, std::sqrt(noise_variance)); // takes sigma not sigma^2
        std::uniform_int_distribution<int> rand_bytes_dist(0, 255);

        const float noisy_symbol_combined_norm = symbol_norm_magnitude * noisy_signal_norm;

        // measure performance
        Timer total_time;
        size_t total_bit_errors = 0;
        size_t total_bits = 0;
        bool is_timeout = false;
        while (true) {
            // generate data
            for (size_t i = 0; i < total_block_bytes; i++) {
                tx_block_bytes[i] = uint8_t(rand_bytes_dist(rand_engine));
            }
            enc->reset();
            encode_data(
                enc,
                tx_block_bytes.data(), tx_block_bytes.size(), 
                output_symbols_float.data(), output_symbols_float.size(),
                1.0f, -1.0f
            );
            // add noise
            for (float& v: output_symbols_float) {
                v += rand_norm_dist(rand_engine);
            }
            // convert to soft decision bits at receiver
            for (size_t i = 0; i < total_block_symbols; i++) {
                const float noisy_bit = output_symbols_float[i];
                const float norm_bit = noisy_bit*noisy_symbol_combined_norm + symbol_norm_mean;
                soft_t soft_bit = soft_t(std::round(norm_bit));
                if (soft_bit > soft_decision_high) soft_bit = soft_decision_high;
                if (soft_bit < soft_decision_low) soft_bit = soft_decision_low;
                output_symbols[i] = soft_bit;
            }
            // traceback
            const size_t total_output_symbols = output_symbols.size();
            vitdec.reset();
            const uint64_t accumulated_error = decoder_t::template update<uint64_t>(
                vitdec, output_symbols.data(), output_symbols.size());
            const uint64_t block_error = accumulated_error + uint64_t(vitdec.get_error());
            vitdec.chainback(rx_block_bytes.data(), total_block_bits, 0u);
            const size_t block_bit_errors = get_total_bit_errors(tx_block_bytes.data(), rx_block_bytes.data(), total_block_bytes);
            total_bit_errors += block_bit_errors;
            total_bits += total_block_bits;
            if (total_bits >= max_generated_bits) break;
            if (total_bit_errors >= args.maximum_error_bits) break;
            if (args.timeout_seconds.has_value()) {
                const float time_elapsed_seconds = float(double(total_time.get_delta()) * 1e-9); 
                if (time_elapsed_seconds > args.timeout_seconds.value()) {
                    is_timeout = true;
                    break;
                }
            }
        }
        const float bit_error_rate = float(total_bit_errors) / float(total_bits);

        results.EbNo_dB.push_back(EbNo_dB);
        results.bit_error_rates.push_back(bit_error_rate);
        results.total_bit_errors.push_back(total_bit_errors);
        results.total_bits.push_back(total_bits);

        auto lock_stderr = std::scoped_lock(mutex_stderr);
        fprintf(stderr, "thread=%zu,name='%s',K=%zu,R=%zu,decode=%s,simd=%s,iter=%zu,EbNo_dB=%.1f,BER=%.3e,timeout=%u\n", 
            thread_id,
            code.name, code.K, code.R, 
            get_decode_type_str(decode_type), get_simd_type_string(simd_type),
            curr_point, EbNo_dB, bit_error_rate, is_timeout
        );
        if (total_bit_errors == 0) break;
        if (curr_point >= args.maximum_data_points) break;
        if (is_timeout) break;
    }

    return results;
}

template <typename T>
void fprintf_list(FILE* fp_out, const char* formatter, tcb::span<const T> list) {
    fprintf(fp_out, "[");
    const size_t N = list.size();
    for (size_t i = 0; i < N; i++) {
        fprintf(fp_out, formatter, list[i]);
        if (i < (N-1)) printf(",");
    }
    fprintf(fp_out, "]");
}

template <size_t K, size_t R, typename code_t>
void print_test_results(
    FILE* fp_out, 
    const Code<K,R,code_t>& code, 
    const DecodeType decode_type, const SIMD_Type simd_type,
    const TestResults& results
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
    fprintf_list(fp_out, "%u", tcb::span<const code_t>(code.G));
    fprintf(fp_out, ",\n");
    fprintf(fp_out, " \"EbNo_dB\": ");
    fprintf_list(fp_out, "%.1f", tcb::span<const float>(results.EbNo_dB));
    fprintf(fp_out, ",\n");
    fprintf(fp_out, " \"ber\": ");
    fprintf_list(fp_out, "%.3e", tcb::span<const float>(results.bit_error_rates));
    fprintf(fp_out, "\n");
    fprintf(fp_out, "}");
}
