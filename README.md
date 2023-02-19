# Introduction
This is a C++ port of Phil Karn's Viterbi decoder which can be found [here](https://github.com/ka9q/libfec).

Modifications include:
- Runtime definable constraint length and code rate
- Runtime definable initial state values and renormalisation threshold
- Reusability of branch table for better memory usage
- Intrinsic support for arbitary constraint lengths if they meet the alignment requirements

NOTE: The provided implementation is not aiming to be the fastest there is. It is designed to be relatively portable and easy to add into a C++ project. Changes are encouraged if your usage requirements demand more performance.

# Intrinsics support
The following intrinsic implementations exist: 
- 16bit error metrics and soft decision values
- 8bit error metrics and soft decision values

| Type | Requirements | Speedup |
| --- | --- | --- |
| Scalar      | K >= 2 | 1x  |
| SSE - 16bit | K >= 5 | 8x  |
| AVX - 16bit | K >= 6 | 16x |
| SSE - 8bit  | K >= 6 | 16x |
| AVX - 8bit  | K >= 7 | 32x |

# Further support
Using the 16bit and 8bit based intrinsics implementations as a guide you can make your own intrinsics implementation that uses a different level of quantization.

You can port the x86 intrinsics implementation to ARM processors since the vector instructions are similiar to NEON.

# Custom modifications
The library uses helper classes which can be replaced with your changes. These include:
- Parity check table
- Aligned malloc wrapper
- Basic functions like min/max/clamp

# Benchmarks
Benchmarks were run using <code>run_benchmarks.cpp</code>.
The input length was adjusted so that the scalar code would take a few seconds.
The benchmark aims to measure the relative speedup using SSE and AVX vectorised intrinsics code.

Values are excluded from the table if the convolutional code cannot be vectorised with intrinsics.

The 16bit and 8bit scalar decoders take similar amounts of time. This means we can compare the speedup from using 8bit and 16bit vectorisations.

## Setup
- <code>./run_benchmark.exe -c \<id\> -M \<mode\> -L \<input_length\> -T \<total_runs\></code>
- Benchmark was executed on an Intel i5-7200U connected to battery power and kept at 3.1GHz.
- 16bit performance was measured with soft decision decoding
-  8bit performance was measured with hard decision decoding
- The code that is executed in the same with soft and hard decision decoding. Only the soft decision values and renormalisation threshold are different.
- No noise was added to the encoded symbols since it is difficult to apply the same amount of noise to a soft decision value and a hard decision value.
- Reasons why a comparison between 16bit and 8bit soft decision decoding is difficult:
    - Renormalisation threshold varies depending on choice of soft decision values, which makes it usage dependent.
    - Difficult to apply the same dB of noise to a 16bit and 8bit soft value due to extreme quantisation.

## 16bit performance
| ID  | Name          |  K  |  R  | SSE  | AVX  | Input size | Total runs |
| --- | ---           | --- | --- | ---  | ---  | ---  | ---  |
|     |               |     |     |  8   | 16   |      |      |
|   0 | Generic       |  3  |  2  | ---- | ---- | ---- | ---- |
|   1 | Generic       |  5  |  2  |  6.5 | ---- | 8192 | 1000 |
|   2 | Voyager       |  7  |  2  |  9.5 | 13.4 | 4096 | 1000 |
|   3 | LTE           |  7  |  3  |  9.6 | 14.8 | 4096 | 1000 |
|   4 | DAB Radio     |  7  |  4  |  9.3 | 15.6 | 2048 | 1000 |
|   5 | CDMA IS-95A   |  9  |  2  | 11.1 | 18.5 | 2048 | 1000 |
|   6 | CDMA 2000     |  9  |  4  | 10.6 | 18.7 | 2048 | 1000 |
|   7 | Cassini       | 15  |  6  | 11.3 | 19.4 |  256 |  100 |

## 8bit performance
| ID  | Name          |  K  |  R  | SSE  | AVX  | Input size | Total runs |
| --- | ---           | --- | --- | ---  | ---  | ---  | ---  |
|     |               |     |     |   16 | 32   |      |      |
|   0 | Generic       |  3  |  2  | ---- | ---- | ---- | ---- |
|   1 | Generic       |  5  |  2  | ---- | ---- | 8192 | 1000 |
|   2 | Voyager       |  7  |  2  | 15.4 | 21.0 | 4096 | 1000 |
|   3 | LTE           |  7  |  3  | 16.1 | 21.9 | 4096 | 1000 |
|   4 | DAB Radio     |  7  |  4  | 16.4 | 22.6 | 4096 | 1000 |
|   5 | CDMA IS-95A   |  9  |  2  | 19.2 | 30.2 | 2048 | 1000 |
|   6 | CDMA 2000     |  9  |  4  | 20.1 | 29.0 | 2048 | 1000 |
|   7 | Cassini       | 15  |  6  | 20.2 | 31.3 |  256 |  100 |

## Analysis of performance
- Both the 16bit and 8bit codes perform better when the contraint length increases
- The 8bit AVX code doesn't scale as well as the 16bit AVX code relative to its SSE counterpart.
- Even with the substandard scaling of 8bit AVX it is still a major speedup.
- Running this code on newer CPUs indicates greater performance gains when using SSE or AVX vectorisation for both 8bit and 16bit.

# Additional notes
- The implementations check if the parameters are valid using assert statements. This is for performance reasons as you usually use known parameters. You will need to add your own runtime checks if you are compiling without asserts.
- The implementations are not ideal for performance since they make some tradeoffs. These include:
    - Using unsigned integer types for error metrics for larger range of error values after renormalisation.
    - Using saturated arithmetic in vectorised code to prevent unwanted overflows/underflows. This incurs up to a 33% speed penalty due to CPI increasing from 0.33 to 0.5 when moving from modular arithmetic.
    - Keeping track of absolute error by accumulating bias adjustments when renormalising.
    - Dynamic allocation for branch table and error metrics using runtime parameters.
- Unsigned 8bit error metrics have severe limitations. These include:
    - Limited range of soft decision values to avoid overflowing past the renormalisation threshold.
    - Higher code rates (such as Cassini) will quickly reach the renormalisation threshold and deteriorate in accuracy. This is because the maximum error for each branch is a multiple of the code rate.
- If you are only interested in hard decision decoding then using 8bit error metrics is the better choice
    - Use hard decision values [-1,+1] of type int8_t
    - Due to the small maximum branch error of 2*R we can set the renormalisation threshold to be quite high. 
    - <code>renormalisation_threshold = UINT8_MAX - (2\*R\*10)</code>
    - This gives similar levels of accuracy to 16bit soft decision decoding but with up to 2x performance due to the usage of 8bit values.
- Depending on your usage requirements changes to the library are absolutely encouraged
- Additionally check out Phil Karn's fine tuned assembly code [here](https://github.com/ka9q/libfec) for the best possible performance 
- This code is not heavily tested and your mileage may vary. This was written for personal usage.

# Useful alternatives
- [Spiral project](https://www.spiral.net/software/viterbi.html) aims to auto-generate high performance code for any input parameters
- [ka9q/libfec](https://github.com/ka9q/libfec) is Phil Karn's original C implementation
