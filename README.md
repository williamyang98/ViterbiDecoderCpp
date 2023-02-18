## Introduction
This is a C++ port of Phil Karn's Viterbi decoder which can be found [here](https://github.com/ka9q/libfec).

Modifications include:
- Runtime definable constraint length and code rate
- Reusability of branch table for better memory usage
- Intrinsic support for arbitary constraint lengths if they meet the alignment requirements

NOTE: The provided implementation is not aiming to be the fastest there is. It is designed to be relatively portable and easy to add into a C++ project. Changes are encouraged if your usage requirements demand more performance.

## Intrinsics support
The intrinsic implementations use uint16_t for errors and int16_t for soft decision values.

| Type | Requirements | Speedup |
| --- | --- | --- |
| Scalar         | K >= 2 | 1x |
| SSE (16 bytes) | K >= 5 | 8x |
| AVX (32 bytes) | K >= 6 | 16x |

## Further support
Using the 16bit based intrinsics implementations as a guide you can make your own intrinsics implementation that uses a different level of quantization.

Example: 8bit intrinsic implementation
- Error type uses uint8_t with range = [0, 255]
- Soft decision type uses int8_t with range = [-10, +10]
- Possible improvements are:
    - SSE: K >= 6, 16x speedup
    - AVX: K >= 7, 32x speedup
- Determining a method to pack decision bits into required format
    - 16bit intrinsics code uses: min_epu16 -> cmpeq_epi16 -> packs_epi16 -> unpack_lo_epi8 -> movemask_epi8
    - This would need to be heavily changed to support 8bit types
    - Refer to ViterbiDecoder_Scalar::bfly for the scalar implementation

You can port the x86 intrinsics implementation to ARM processors since the vector instructions are similiar to NEON.

## Custom modifications
The library uses helper classes which can be replaced with your changes. These include:
- Bit count table 
- Parity check table
- Aligned malloc wrapper
- Basic functions like min/max/clamp

## Additional notes
- The implementations check if the parameters are valid using assert statements. This is for performance reasons as you usually use known parameters. You will need to add your own runtime checks if you are compiling without asserts.
- The implementations are not ideal for performance since they make some tradeoffs. These include:
    - Using uint16_t for error metrics for better renormalisation
    - Keeping track of absolute error by accumulating bias adjustments when renormalising
    - Dynamic allocation for branch table and error metrics using runtime parameters
    - Virtual functions for runtime substitution of scalar or SIMD decoders
- Depending on your usage requirements changes to the library are absolutely encouraged
- Additionally check out Phil Karn's fine tuned assembly code [here](https://github.com/ka9q/libfec) for the best possible performance 
- This code is not heavily tested and your mileage may vary. This was written for personal usage.

## Useful alternatives
- [Spiral project](https://www.spiral.net/software/viterbi.html) aims to auto-generate high performance code for any input parameters
- [ka9q/libfec](https://github.com/ka9q/libfec) is Phil Karn's original C implementation

## Benchmarks
Benchmarks were run using <code>run_benchmarks.cpp</code>.
The input length was adjusted so that the scalar code would take a few seconds.
The benchmark aims to measure the relative speedup using SSE and AVX vectorised intrinsics code.

Values are excluded from the table if the convolutional code cannot be vectorised with intrinsics.

<code>./run_benchmark.exe -c \<id\> -L \<input_length\> -T \<total_runs\></code>

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