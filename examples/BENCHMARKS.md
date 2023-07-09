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
