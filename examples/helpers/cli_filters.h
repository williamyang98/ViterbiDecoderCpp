#pragma once
#include <cctype>
#include <vector>
#include <optional>
#include <stdio.h>
#include <stdlib.h>
#include "helpers/common_codes.h"
#include "helpers/simd_type.h"
#include "helpers/decode_type.h"

struct CLI_Filters {
    std::optional<size_t> code_index = std::nullopt;
    std::optional<DecodeType> decode_type = std::nullopt;
    std::optional<SIMD_Type> simd_type = std::nullopt;
    bool allow_code_index(size_t i) {
        if (code_index == std::nullopt) return true;
        return code_index.value() == i;
    }
    bool allow_decode_type(DecodeType type) {
        if (decode_type == std::nullopt) return true;
        return decode_type.value() == type;
    }
    bool allow_simd_type(SIMD_Type type) {
        if (simd_type == std::nullopt) return true;
        return simd_type.value() == type;
    }
};

struct CLI_Simd_Option {
    std::string arg;
    SIMD_Type type;
};
static auto cli_simd_options = []() {
    std::vector<CLI_Simd_Option> values;
    for (auto type: SIMD_Type_List) {
        const char* str = get_simd_type_string(type);
        std::string val = std::string(str);
        for (auto& c: val) { c = std::tolower(c); }
        values.push_back(CLI_Simd_Option { val, type });
    }
    return values;
}();

struct CLI_Decode_Option {
    std::string arg;
    DecodeType type;
};
static auto cli_decode_options = []() {
    std::vector<CLI_Decode_Option> values;
    for (auto type: Decode_Type_List) {
        const char* str = get_decode_type_str(type);
        std::string val = std::string(str);
        for (auto& c: val) { c = std::tolower(c); };
        values.push_back(CLI_Decode_Option { val, type });
    }
    return values;
}();

static void cli_print_codes() {
    constexpr size_t max_name_length = 16;
    fprintf(stderr, "ID | %*s |  K  R | Coefficients\n", int(max_name_length), "Name");
    size_t code_index = 0;
    FOR_COMMON_CODES({
        const auto& code = it;
        fprintf(stderr, "%2zu | %*s | %2zu %2zu | ", code_index, int(max_name_length), code.name, code.K, code.R);
        fprintf(stderr, "[");
        bool is_first = true;
        for (size_t i = 0u; i < code.G.size(); i++) {
            if (!is_first) fprintf(stderr, ",");
            fprintf(stderr, "%u", code.G[i]);
            is_first = false;
        }
        fprintf(stderr, "]");
        fprintf(stderr, "\n");
        code_index++;
    });
}

static std::optional<SIMD_Type> cli_get_simd_type(const char* arg) {
    for (const auto& e: cli_simd_options) {
        if (e.arg.compare(arg) == 0) return std::optional(e.type);
    }
    return std::nullopt;
}

static std::optional<DecodeType> cli_get_decode_type(const char* arg) {
    for (const auto& e: cli_decode_options) {
        if (e.arg.compare(arg) == 0) return std::optional(e.type);
    }
    return std::nullopt;
}


#define CLI_FILTERS_GETOPT_STRING "c:d:s:l"

static void cli_filters_print_usage() {
    fprintf(stderr, 
        "    [-c <code_index> (default: None)]\n");

    bool is_first = true;
    fprintf(stderr, 
        "    [-d <decode_type> (default: None)]\n"
        "        options: ["
    );
    for (const auto& opt: cli_decode_options) {
        if (!is_first) fprintf(stderr, ",");
        fprintf(stderr, "%*s", int(opt.arg.size()), opt.arg.c_str());
        is_first = false;
    }
    fprintf(stderr, "]\n");

    fprintf(stderr,
        "    [-s <simd_type> (default: None)]\n"
        "        options: ["
    );
    is_first = true;
    for (const auto& opt: cli_simd_options) {
        if (!is_first) fprintf(stderr, ",");
        fprintf(stderr, "%*s", int(opt.arg.size()), opt.arg.c_str());
        is_first = false;
    };
    fprintf(stderr, "]\n");
    fprintf(stderr, 
        "    [-l List all available codes ]\n");
}

enum class CLI_Filters_Getopt_Result {
    NONE,
    ERROR_PARSE,
    SUCCESS_PARSE,
    SUCCESS_EXIT,
};
// -1 = do nothing, 1 = error parsing, 0 = 
static CLI_Filters_Getopt_Result cli_filters_parse_getopt(
    CLI_Filters& filters, int opt, const char* optarg, const char* argv0
) {
    using R = CLI_Filters_Getopt_Result;
    switch (opt) {
        case 'c': {
            const int index = atoi(optarg);
            if ((index < 0) || (index >= COMMON_CODES.N)) {
                fprintf(stderr, "Code index must be between 0 and %zu: %s\n", COMMON_CODES.N-1, optarg);
                fprintf(stderr, "Run '%s -l' for list of codes\n", argv0);
                return R::ERROR_PARSE;
            }
            filters.code_index = std::optional(size_t(index));
            return R::SUCCESS_PARSE;
        };
        case 'd': {
            filters.decode_type = cli_get_decode_type(optarg);
            if (filters.decode_type == std::nullopt) {
                fprintf(stderr, "Invalid option for decode type: '%s'\n", optarg);
                fprintf(stderr, "Run '%s -h' for list of valid decode types for -d\n", argv0);
                return R::ERROR_PARSE;
            }
            return R::SUCCESS_PARSE;
        };
        case 's': {
            filters.simd_type = cli_get_simd_type(optarg);
            if (filters.simd_type == std::nullopt) {
                fprintf(stderr, "Invalid option for simd type: '%s'\n", optarg);
                fprintf(stderr, "Run '%s -h' for list of valid simd types for -s\n", argv0);
                return R::ERROR_PARSE;
            }
            return R::SUCCESS_PARSE;
        };
        case 'l': {
            cli_print_codes();
            return R::SUCCESS_EXIT;
        };
        default:
            return R::NONE;
    }
    return R::NONE;
}

