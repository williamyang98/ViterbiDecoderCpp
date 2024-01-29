import argparse
import json
import math
from enum import Enum
# dependencies
import matplotlib.pyplot as plt

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
        if type.name == x: 
            return type
    raise Exception(f"invalid decode type '{x}'")

def get_simd_type(x: str) -> SimdType:
    for type in SimdType:
        if type.name == x: 
            return type
    raise Exception(f"invalid simd type '{x}'")

class Sample:
    def __init__(self, data):
        self.name = data["name"]
        self.decode_type = get_decode_type(data["decode_type"])
        self.simd_type = get_simd_type(data["simd_type"])
        self.K = data["K"]
        self.R = data["R"]
        self.EbNo_dB = []
        self.ber = []
        # filter for log y-axis
        for (x,y) in zip(data["EbNo_dB"], data["ber"]):
            if y == 0.0:
                continue
            self.EbNo_dB.append(x)
            self.ber.append(y)

def main():
    parser = argparse.ArgumentParser(prog="plot_snr_ber", description="Plot SNR vs BER json data")
    parser.add_argument("filename", help="Output from run_snr_ber.cpp")
    args = parser.parse_args()

    with open(args.filename, "r") as fp:
        json_text = fp.read()
 
    X_in = json.loads(json_text)
    all_samples = [Sample(x) for x in X_in]

    list_keys = set(((s.name, s.K, s.R) for s in all_samples))
    list_keys = list(sorted(list_keys, key=lambda s: s[1]*s[2]))

    decode_type_colour_map = plt.get_cmap("tab10")
    decode_type_colours = {t:decode_type_colour_map(i) for i,t in enumerate(DecodeType)}
 
    # plot in groups of name->simd_type->decode_type
    name_groups = []
    for (name, K, R) in list_keys:
        samples = [s for s in all_samples if s.name == name] 
        simd_groups = []
        for simd_type in SimdType:
            simd_samples = [s for s in samples if s.simd_type == simd_type]
            if len(simd_samples) == 0:
                continue
            simd_groups.append(simd_samples)
        if len(simd_groups) == 0:
            continue
        name_groups.append(simd_groups)
 
    for simd_group in name_groups:
        fig = plt.figure(1)
        # determine ticks for all subplots
        xticks = []
        yticks = []
        for samples in simd_group:
            for s in samples:
                xticks.extend(s.EbNo_dB)
                yticks.extend(s.ber)
        xticks = list(sorted(set(xticks)))
        yticks_max = math.ceil(math.log10(max(yticks)))
        yticks_min = math.floor(math.log10(min(yticks)))
        yticks = [10**i for i in range(yticks_min, yticks_max+1)]
        # render subplots
        total_rows = len(simd_group)
        ax0 = None
        for row_id, samples in enumerate(simd_group):
            ax = fig.add_subplot(total_rows, 1, row_id+1, sharex=ax0, sharey=ax0)
            if row_id == 0:
                ax0 = ax
            for s in samples:
                line_colour = decode_type_colours[s.decode_type]
                ax.semilogy(s.EbNo_dB, s.ber, label=s.decode_type.name.lower(), marker=".", color=line_colour)
            ax.set_title(f"{s.simd_type.name}", fontsize=9)
            ax.grid(True, which="both")
            ax.legend(loc="lower left") # BER is highest for lower Eb/No values
            ax.set_xticks(xticks, minor=True)
            ax.set_yticks(yticks, minor=True)
            ax.set_ylabel("Bit error rate")
            if row_id == total_rows-1:
                ax.set_xlabel("Eb/No (dB)")
            else:
                ax.tick_params(labelbottom=False)

        plt.suptitle(f"{s.name} (K={s.K},R={s.R})")
        plt.show()

if __name__ == '__main__':
    main()
