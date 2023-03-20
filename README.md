# Introduction
![x86-windows-test](https://github.com/FiendChain/ViterbiDecoderCpp/actions/workflows/x86-windows.yml/badge.svg)
![x86-linux-test](https://github.com/FiendChain/ViterbiDecoderCpp/actions/workflows/x86-linux.yml/badge.svg)
![arm-linux-test](https://github.com/FiendChain/ViterbiDecoderCpp/actions/workflows/arm-linux.yml/badge.svg)

This is a C++ port of Phil Karn's Viterbi decoder which can be found [here](https://github.com/ka9q/libfec).

**This is a header only library. Just copy and paste the header files to your desired location.**

See <code>examples/run_simple.cpp</code> for a common usage scenario.

Modifications include:
- Templated code for creating decoders of any code rate and constraint length
- Branch table can be specified at runtime and shared between multiple decoders
- Initial error values and renormalisation threshold can be specified
- Vectorisation using intrinsics for arbitary constraint lengths (where possible) for significant speedups

Performance is similar to Phil Karn's original C implementation for provided decoders.

Heavy templating is used for better performance. Compared to code that uses constraint length (K) and code rate (R) as runtime parameters [here](https://github.com/FiendChain/ViterbiDecoderCpp/tree/44cdd3c0a38a748a7084edeff859cf4d54ac911a), the templated version is up to 50% faster. This is because the compiler can perform more optimisations if the constraint length and code rate are known ahead of time.

# Intrinsics support
For x86 processors AVX2 or SSE4.1 is required for vectorisation.

For arm processors aarch64 is required for vectorisation.

The following intrinsic implementations exist: 
- 16bit error metrics and soft decision values
- 8bit error metrics and soft decision values

Each vectorisaton type requires the convolution code to have a minimum constraint length (K)

| Type | Width | Kmin | Speedup |
| --- | --- | --- | --- |
| Scalar     |       | 2 | 1x  |
| x86 SSE4.1 | 16bit | 5 | 8x  |
| x86 SSE4.1 | 8bit  | 6 | 16x |
| x86 AVX2   | 16bit | 6 | 16x |
| x86 AVX2   | 8bit  | 7 | 32x |
| ARM Neon   | 16bit | 5 | 8x  |
| ARM Neon   | 8bit  | 6 | 16x |

Benchmarks show that the vectorised decoders have significiant speedups that can approach or supercede the theoretical values.

Using the 16bit and 8bit based intrinsics implementations as a guide you can make your own intrinsics implementation.

# Additional notes
- **Significant performance improvements can be achieved with using [offset binary](https://en.wikipedia.org/wiki/Offset_binary)**
    - Soft decision values take the form of offset binary given by: 0 to N
    - The branch table needs values: 0 and N
    - Instead of performing a subtract then absolute, you can use an XOR operation which behaves like conditional negation
    - Refer to the [original Phil Karn code](https://github.com/ka9q/libfec/blob/7c6706fb969c3f8fe6ec7778b2472762e0d88acc/viterbi615_sse2.c#L128) for this improvement
    - Explanation with example
        - Branch table has values: 0 or 255
        - Soft decision values in offset binary: 0 to 255
        - Consider a soft decision value of x
        - If branch value is 0, XOR will return x
        - If branch value is 255, XOR will return 255-x
- Performance improvements can be achieved with using signed integer types
    - Using signed integer types allows for the use of modular arithmetic instead of saturated arithmetic. This can provide a up to a 33% speed boost due to CPI decreasing from 0.5 to 0.33.
    - Unsigned integer types are used since they increase the range of error values after renormalisation, and saturated arithmetic will prevent overflows/underflows.
- The implementations uses template parameters and static asserts to: 
    - Check if the provided constraint length and code rate meet the vectorisation requirements
    - Generate aligned data structures and provide the compiler more information about the decoder for better optimisation
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
- This code is not considered heavily tested and your mileage may vary. This was written for personal usage.

# Useful alternatives
- [Spiral project](https://www.spiral.net/software/viterbi.html) aims to auto-generate high performance code for any input parameters
- [ka9q/libfec](https://github.com/ka9q/libfec) is Phil Karn's original C implementation

# Benchmarks
- <code>./run_benchmark.exe -c \<id\> -M \<mode\> -L \<input_length\> -T \<total_runs\></code>
- The goal is to measure the relative speedup using SSE and AVX vectorised intrinsics code. 
- 16bit performance was measured with soft decision decoding
-  8bit performance was measured with hard decision decoding
- No noise was added to the encoded symbols
- Values are excluded from the table if it cannot be vectorised
- Time values are measured in seconds
- Speed up multiplier over scalar code is provided inside round brackets when vectorisation is possible

## System Setup
- Executed on an Intel i5-7200U connected to battery power and kept at 3.1GHz
- MSVC: 19.32.31332 for x64
- GCC: 11.3.0-1 ubuntu-22.04.2-LTS for WSL

## Results
### 1. MSVC + 16bit
<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>K</th>
            <th>R</th>
            <th>Input size</th>
            <th>Total runs</th>
            <th>Scalar</th>
            <th>SSE</th>
            <th>AVX</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td><td>Generic</td><td>3</td><td>2</td><td>1024</td><td>50000</td>
            <td>5.227</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>1</td><td>Generic</td><td>5</td><td>2</td><td>1024</td><td>10000</td>
            <td>3.731</td>
            <td>0.433 (8.6)</td>
            <td></td>
        </tr>
        <tr>
            <td>2</td><td>Voyager</td><td>7</td><td>2</td><td>1024</td><td>5000</td>
            <td>7.029</td>
            <td>0.680 (10.3)</td>
            <td>0.456 (15.4)</td>
        </tr>
        <tr>
            <td>3</td><td>LTE</td><td>7</td><td>3</td><td>1024</td><td>2500</td>
            <td>4.005</td>
            <td>0.435 (9.2)</td>
            <td>0.280 (14.3)</td>
        </tr>
        <tr>
            <td>4</td><td>DAB Radio</td><td>7</td><td>4</td><td>1024</td><td>2500</td>
            <td>4.510</td>
            <td>0.422 (10.7)</td>
            <td>0.274 (16.4)</td>
        </tr>
        <tr>
            <td>5</td><td>CDMA IS-95A</td><td>9</td><td>2</td><td>1024</td><td>1000</td>
            <td>5.406</td>
            <td>0.497 (10.9)</td>
            <td>0.285 (19.0)</td>
        </tr>
        <tr>
            <td>6</td><td>CDMA 2000</td><td>9</td><td>4</td><td>1024</td><td>1000</td>
            <td>6.745</td>
            <td>0.735 (9.2)</td>
            <td>0.371 (18.2)</td>
        </tr>
        <tr>
            <td>7</td><td>Cassini</td><td>15</td><td>6</td><td>256</td><td>100</td>
            <td>13.472</td>
            <td>1.274 (10.6)</td>
            <td>0.865 (15.6)</td>
        </tr>
    </tbody>
</table>

### 2. GCC + 16bit
<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>K</th>
            <th>R</th>
            <th>Input size</th>
            <th>Total runs</th>
            <th>Scalar</th>
            <th>SSE</th>
            <th>AVX</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td><td>Generic</td><td>3</td><td>2</td><td>1024</td><td>50000</td>
            <td>3.643</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>1</td><td>Generic</td><td>5</td><td>2</td><td>1024</td><td>10000</td>
            <td>2.972</td>
            <td>0.397 (7.5)</td>
            <td></td>
        </tr>
        <tr>
            <td>2</td><td>Voyager</td><td>7</td><td>2</td><td>1024</td><td>5000</td>
            <td>0.766</td>
            <td>0.645 (1.2)</td>
            <td>0.369 (2.1)</td>
        </tr>
        <tr>
            <td>3</td><td>LTE</td><td>7</td><td>3</td><td>1024</td><td>2500</td>
            <td>0.376</td>
            <td>0.311 (1.2)</td>
            <td>0.196 (1.9)</td>
        </tr>
        <tr>
            <td>4</td><td>DAB Radio</td><td>7</td><td>4</td><td>1024</td><td>2500</td>
            <td>0.412</td>
            <td>0.340 (1.2)</td>
            <td>0.221 (1.9)</td>
        </tr>
        <tr>
            <td>5</td><td>CDMA IS-95A</td><td>9</td><td>2</td><td>1024</td><td>1000</td>
            <td>5.033</td>
            <td>0.420 (12.0)</td>
            <td>0.238 (21.1)</td>
        </tr>
        <tr>
            <td>6</td><td>CDMA 2000</td><td>9</td><td>4</td><td>1024</td><td>1000</td>
            <td>5.936</td>
            <td>0.628 (9.5)</td>
            <td>0.378 (15.7)</td>
        </tr>
        <tr>
            <td>7</td><td>Cassini</td><td>15</td><td>6</td><td>256</td><td>100</td>
            <td>11.364</td>
            <td>1.090 (10.4)</td>
            <td>0.617 (18.4)</td>
        </tr>
    </tbody>
</table>

### 3. MSVC + 8bit
<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>K</th>
            <th>R</th>
            <th>Input size</th>
            <th>Total runs</th>
            <th>Scalar</th>
            <th>SSE</th>
            <th>AVX</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td><td>Generic</td><td>3</td><td>2</td><td>1024</td><td>50000</td>
            <td>5.259</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>1</td><td>Generic</td><td>5</td><td>2</td><td>1024</td><td>10000</td>
            <td>3.367</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>2</td><td>Voyager</td><td>7</td><td>2</td><td>1024</td><td>5000</td>
            <td>7.020</td>
            <td>0.501 (14.0)</td>
            <td>0.301 (23.4)</td>
        </tr>
        <tr>
            <td>3</td><td>LTE</td><td>7</td><td>3</td><td>1024</td><td>2500</td>
            <td>3.966</td>
            <td>0.232 (17.1)</td>
            <td>0.156 (25.4)</td>
        </tr>
        <tr>
            <td>4</td><td>DAB Radio</td><td>7</td><td>4</td><td>1024</td><td>2500</td>
            <td>4.469</td>
            <td>0.279 (16.0)</td>
            <td>0.198 (22.6)</td>
        </tr>
        <tr>
            <td>5</td><td>CDMA IS-95A</td><td>9</td><td>2</td><td>1024</td><td>1000</td>
            <td>5.313</td>
            <td>0.253 (21.0)</td>
            <td>0.195 (27.3)</td>
        </tr>
        <tr>
            <td>6</td><td>CDMA 2000</td><td>9</td><td>4</td><td>1024</td><td>1000</td>
            <td>7.268</td>
            <td>0.322 (22.6)</td>
            <td>0.233 (31.2)</td>
        </tr>
        <tr>
            <td>7</td><td>Cassini</td><td>15</td><td>6</td><td>256</td><td>100</td>
            <td>14.104</td>
            <td>0.692 (20.4)</td>
            <td>0.450 (31.3)</td>
        </tr>
    </tbody>
</table>

### 4. GCC + 8bit
<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>K</th>
            <th>R</th>
            <th>Input size</th>
            <th>Total runs</th>
            <th>Scalar</th>
            <th>SSE</th>
            <th>AVX</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td><td>Generic</td><td>3</td><td>2</td><td>1024</td><td>50000</td>
            <td>3.819</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>1</td><td>Generic</td><td>5</td><td>2</td><td>1024</td><td>10000</td>
            <td>4.504</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>2</td><td>Voyager</td><td>7</td><td>2</td><td>1024</td><td>5000</td>
            <td>5.953</td>
            <td>0.339 (17.6)</td>
            <td>0.269 (22.2)</td>
        </tr>
        <tr>
            <td>3</td><td>LTE</td><td>7</td><td>3</td><td>1024</td><td>2500</td>
            <td>3.335</td>
            <td>0.184 (18.1)</td>
            <td>0.147 (22.7)</td>
        </tr>
        <tr>
            <td>4</td><td>DAB Radio</td><td>7</td><td>4</td><td>1024</td><td>2500</td>
            <td>3.680</td>
            <td>0.197 (18.7)</td>
            <td>0.142 (26.0)</td>
        </tr>
        <tr>
            <td>5</td><td>CDMA IS-95A</td><td>9</td><td>2</td><td>1024</td><td>1000</td>
            <td>5.198</td>
            <td>0.231 (22.5)</td>
            <td>0.149 (34.9)</td>
        </tr>
        <tr>
            <td>6</td><td>CDMA 2000</td><td>9</td><td>4</td><td>1024</td><td>1000</td>
            <td>6.236</td>
            <td>0.293 (21.3)</td>
            <td>0.172 (36.3)</td>
        </tr>
        <tr>
            <td>7</td><td>Cassini</td><td>15</td><td>6</td><td>256</td><td>100</td>
            <td>11.622</td>
            <td>0.571 (20.3)</td>
            <td>0.344 (33.8)</td>
        </tr>
    </tbody>
</table>

## Analysis of performance
- The 8bit vectorised decoders are faster than the 16bit vectorised decoders of up to 2 times
- GCC produces significantly faster code than MSVC
- GCC will automatically vectorise 16bit scalar code for constraint lengths of 7. However the manual SSE vector code out performs it by 1.2x.
