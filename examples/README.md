# Introduction
Example programs which use the viterbi decoder library.

## Building programs
1. ```cd examples```
2. Configure cmake: ```cmake . -B build --preset windows-msvc -DCMAKE_BUILD_TYPE=Release```
3. Build: ```cmake --build build --config Release```

Change preset for your specific compiler. Refer to ```CMakePresets.json``` for example presets.

## Programs
| Name | Description |
| --- | --- |
| run_simple            | A simple and common decoder use pattern |
| run_tests             | Runs test suite on all decoder combinations |
| run_benchmark         | Runs benchmark to compare performance between vectorisations |
| run_punctured_decoder | Implementation of DAB radio punctured decoding |
| run_snr_ber           | Measures bit error rate vs SNR for all decoders and prints to stdout |

### Run tests
1. ```./build/run_tests.exe```

### Plot performance of decoders
1. ```./build/run_snr_ber.exe > ./data_snr_ber_0.txt```
2. ```pip install matplotlib```
3. ```python ./plot_snr_ber.py ./data_snr_ber_0.txt```

**NOTE**: soft_8 decoders for high code rates will overflow for scalar implementations due to non saturating arithmetic.

### Run benchmark
1. ```./build/run_benchmark.exe > ./data_benchmark_0.txt```
2. ```pip install numpy```
3. ```python ./parse_benchmark.py ./data_benchmark_0.txt```

To compare benchmarks use:

```diff -y <(python ./parse_benchmark.py ./0.txt) <(python ./parse_benchmark.py ./1.txt)```
