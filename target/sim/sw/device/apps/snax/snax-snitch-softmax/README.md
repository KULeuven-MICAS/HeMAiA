# Snitch fixed-point softmax test

This app computes independent row-wise softmax vectors on the Snitch compute
cores. The DMA core stages the input, lookup table, and references in TCDM.
Compute core `i` handles rows `i`, `i + num_compute_cores`, and so on.

The runtime kernel contains no floating-point operations. An input integer `x`
represents `x / 2^input_frac_bits`. For each row it computes:

1. `max = max(input)`.
2. `delta = max - input`, which is in `[0, 255]` for int8 input.
3. `exp_value = exp_lut[delta]`, stored in fixed-point Q15 by default.
4. `sum = sum(exp_value)`.
5. `output = round(exp_value * output_scale / sum)`.

The exponential LUT is generated offline; the shared LUT scale cancels during
normalization. Output entries are rounded independently, so their integer sum
can differ slightly from `output_scale`.

## Parameters

Edit `data/params.hjson` and rebuild. The most important fields are:

- `softmax_rows`: number of independent vectors; default 64.
- `softmax_cols`: reduction/vector length; default 64.
- `input_frac_bits`: fixed-point interpretation of each int8 score; default 4.
- `exp_frac_bits`: exponential LUT precision; default 15.
- `output_scale`: quantized representation of probability 1.0; default 255.
- `target_cluster`: cluster that executes the test; default 1.

The generator creates random data plus uniform, peaked, and ramp corner cases.
It emits both a bit-exact integer golden result and an independently calculated
floating-point reference.

## Build

Regenerate only the parameterized data from this directory with:

```sh
make clean-data
make -C data
```

The app is registered in `target/sim/sw/host/apps/offload`, so the normal
repository software build generates `build/origin.ld` and links the device
binary. After that origin file exists, `make all` in this directory also builds
the app directly.

For integration after `QK^T`, set `input_frac_bits` to match the real scale of
the quantized attention scores, including the intended `1/sqrt(d_k)` factor.
The kernel emits unsigned probabilities. If the following GEMM accepts signed
int8 only, encode probability `p` as `p - 128` and compensate with an input zero
point of `-128`.
