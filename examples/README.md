# Introduction
Example programs which use the viterbi decoder library.

## Building programs
1. <code>cd examples</code>
2. <code>cmake . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release</code>
3. <code>build/run_tests.exe</code>

## Programs
| Name | Description |
| --- | --- |
| run_simple            | A simple and common decoder use pattern |
| run_tests             | Runs test suite on all decoder combinations |
| run_benchmark         | Runs benchmark to compare performance between vectorisations |
| run_decoder           | Runs any arbitary decoder |
| run_punctured_decoder | Implementation of DAB radio punctured decoding |
