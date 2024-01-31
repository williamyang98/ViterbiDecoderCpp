import argparse
import json
import math
from enum import Enum
# dependencies
import numpy as np

class DecodeType(Enum):
    SOFT16 = 0
    SOFT8 = 1
    HARD8 = 2

class SimdType(Enum):
    SCALAR = 0
    SIMD_SSE = 1
    SIMD_AVX = 2
    SIMD_NEON = 3

def get_decode_type(x: str) -> DecodeType:
    for type in DecodeType:
        if type.name.lower() == x.lower(): 
            return type
    raise Exception(f"invalid decode type '{x}'")

def get_simd_type(x: str) -> SimdType:
    for type in SimdType:
        if type.name.lower() == x.lower(): 
            return type
    raise Exception(f"invalid simd type '{x}'")

class Sample:
    def __init__(self, data):
        self.name = data["name"]
        self.decode_type = get_decode_type(data["decode_type"])
        self.simd_type = get_simd_type(data["simd_type"])
        self.K = data["K"]
        self.R = data["R"]
        self.G = data["G"]
        self.total_input_bits = data["total_input_bits"]
        self.total_symbols = data["total_symbols"]
        self.update_symbols_ns = np.array(data["update_symbols_ns"])
        self.chainback_bit_ns = np.array(data["chainback_bits_ns"])
        self.symbol_rate = self.total_symbols / (self.update_symbols_ns*1e-9)
        self.chainback_rate = self.total_input_bits / (self.chainback_bit_ns*1e-9)

SCALE_SUFFIXES = [(1e12,"tera"),(1e9,"giga"),(1e6,"mega"),(1e3,"kilo"),(1e0,""),(1e-3,"milli"),(1e-6,"micro"),(1e-9,"nano")]
def get_scale_suffix(x: float) -> (float, str):
    for scale, prefix in SCALE_SUFFIXES:
        if x > scale:
            return (scale, prefix)
    return (1,"")

def main():
    parser = argparse.ArgumentParser(
        prog="parse_benchmark", 
        description="Parse and compare benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("filename", help="Output from run_benchmark.cpp")
    parser.add_argument("--filter-code", help="Filter for specific code by name", default=[], nargs='+')
    parser.add_argument("--filter-decode", help="Filter for specific decoder type", choices=[e.name.lower() for e in DecodeType], default=[], nargs='+')
    parser.add_argument("--filter-simd", help="Filter for specific simd type", choices=[e.name.lower() for e in SimdType], default=[], nargs='+')
    parser.add_argument("--list-codes", help="List all codes in file", action='store_true')
    args = parser.parse_args()
 
    # parse
    with open(args.filename, "r") as fp:
        json_text = fp.read()
    json_data = json.loads(json_text)
    all_samples = [Sample(x) for x in json_data]

    if args.list_codes:
        samples = {s.name:s for s in all_samples}.values()
        max_name_length = max((len(s.name) for s in samples))
        print(f" {'Name'.ljust(max_name_length)} |  K  R | Coefficients")
        for s in sorted(samples, key=lambda s: (2**s.K)*s.R):
            print(f" {s.name.ljust(max_name_length)} | {s.K:2d} {s.R:2d} | {s.G}")
        return

    # filter
    filter_decode = [get_decode_type(x) for x in args.filter_decode]
    filter_simd = [get_simd_type(x) for x in args.filter_simd]
    def filter_samples(s):
        if filter_simd and not s.simd_type in filter_simd:
            return False
        if filter_decode and not s.decode_type in filter_decode:
            return False
        if args.filter_code and not s.name in args.filter_code:
            return False
        return True
    all_samples = list(filter(filter_samples, all_samples))

    # print in groups of name->decode_type->simd_type
    sorted_keys = set(((s.name, s.K, s.R) for s in all_samples))
    sorted_keys = list(sorted(sorted_keys, key=lambda s: (2**s[1]-1)*s[2]))
    name_groups = []
    for (name, K, R) in sorted_keys:
        samples = [s for s in all_samples if s.name == name] 
        decode_groups = []
        for decode_type in DecodeType:
            decode_samples = [s for s in samples if s.decode_type == decode_type]
            if len(decode_samples) == 0:
                continue
            decode_groups.append(decode_samples)
        if len(decode_groups) == 0:
            continue
        name_groups.append(decode_groups)

    for decode_groups in name_groups:
        for samples in decode_groups:
            samples = list(sorted(samples, key=lambda s: s.simd_type.value))
            scalar_mean_symbol_rate = None
            scalar_mean_chainback_rate = None
            s = samples[0]
            print(f"name='{s.name}',K={s.K},R={s.R},decode={s.decode_type.name}")
            for s in samples:
                mean_symbol_rate = np.mean(s.symbol_rate)
                mean_chainback_rate = np.mean(s.chainback_rate)
                std_symbol_rate = np.std(s.symbol_rate)
                std_chainback_rate = np.std(s.chainback_rate)
                if not scalar_mean_symbol_rate is None:
                    ratio_update = mean_symbol_rate/scalar_mean_symbol_rate
                    ratio_chainback = mean_chainback_rate/scalar_mean_chainback_rate
                    postfix_update = f"(x{ratio_update:.2f})"
                    postfix_chainback = f"(x{ratio_chainback:.2f})"
                else:
                    postfix_update = ""
                    postfix_chainback = ""

                scale, prefix = get_scale_suffix(mean_symbol_rate) 
                str_update = f"{mean_symbol_rate/scale:.2f} ± {std_symbol_rate/scale:.2f} {prefix}"
                scale, prefix = get_scale_suffix(mean_chainback_rate) 
                str_chainback = f"{mean_chainback_rate/scale:.2f} ± {std_chainback_rate/scale:.2f} {prefix}"

                print(f"simd={s.simd_type.name.lower()},samples={len(s.update_symbols_ns)}")
                print(f" update    = {str_update}symbols/s {postfix_update}")
                print(f" chainback = {str_chainback}bits/s {postfix_chainback}")
                if s.simd_type == SimdType.SCALAR:
                    scalar_mean_symbol_rate = mean_symbol_rate
                    scalar_mean_chainback_rate = mean_chainback_rate
            print()

if __name__ == '__main__':
    main()
