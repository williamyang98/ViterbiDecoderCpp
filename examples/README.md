# Introduction
Example programs which use the viterbi decoder library.

## Building programs
1. ```cd examples```
2. ```cmake . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release```

## Programs
| Name | Description |
| --- | --- |
| run_simple            | A simple and common decoder use pattern |
| run_tests             | Runs test suite on all decoder combinations |
| run_benchmark         | Runs benchmark to compare performance between vectorisations |
| run_punctured_decoder | Implementation of DAB radio punctured decoding |
| run_snr_ber           | Measures bit error rate vs SNR for all decoders and prints to stdout |

## Run tests
1. ```./build/run_tests.exe```

## Plot performance of decoders
1. ```./build/run_snr_ber.exe | tee ./data_snr_ber_0.txt```
2. ```pip install matplotlib```
3. ```python ./plot_snr_ber.py ./data_snr_ber_0.txt```

**NOTE**: soft_8 decoders for high code rates will overflow for scalar implementations due to non saturating arithmetic.
