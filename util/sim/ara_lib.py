#!/usr/bin/env python3
"""Shared test-data + golden generator for the Ara FP32 RVV host kernels.

Single source of truth for both:
  * the decomposed ara_<kernel> timing-sweep apps -- each
    target/sw/host/apps/host_only/single_chip/ara_<kernel>/Makefile runs
    ``ara_lib.py <kernel>`` to emit its include/op_test_data.h (the big input
    arrays + that kernel's golden, so the sweep can check its result), and
  * the ci_ara correctness regression -- ci_ara/include/gen_op_data.py imports
    ``emit_ci_header`` so it draws identical inputs from the same PRNG/ranges and
    uses the same golden formulas.

Golden values are computed with exact Python math; the kernels use a polynomial
exp approximation, so the C-side checks apply a loose tolerance for exp-based ops
(see ARA_TOL in each ara_<kernel>/src/main.c).
"""
import math
import random
import sys

SEED = 42
N_BIG = 4096                 # big arrays for the timing sweep
SIZES = (64, 256, 1024, 4096)  # sweep sizes -- must match main.c timing_sizes


# ---------------------------------------------------------------------------
# Inputs (seeded, identical sequence for any length -> small N is a prefix)
# ---------------------------------------------------------------------------

def _inputs(n, seed=SEED):
    rng = random.Random(seed)
    a = [round(rng.uniform(0.3, 4.0), 4) for _ in range(n)]   # >0: exp/sqrt/recip safe
    b = [round(rng.uniform(0.5, 3.0), 4) for _ in range(n)]   # >0: no div-by-~0
    m = [round(rng.uniform(-5.0, 5.0), 4) for _ in range(n)]  # signed
    return a, b, m


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Golden formulas (match host_kernel_lib.h)
# ---------------------------------------------------------------------------

BINARY = {  # (a[i], b[i]) -> out[i]
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
    "max": lambda a, b: max(a, b), "min": lambda a, b: min(a, b),
    "silu_mul": lambda a, b: (a * _sigmoid(a)) * b,   # silu(gate)*up, gate=a, up=b
}
UNARY = {  # name -> (input array key, x -> out)
    "exp": ("a", math.exp), "sigmoid": ("m", _sigmoid), "sqrt": ("a", math.sqrt),
    "relu": ("m", lambda x: max(0.0, x)), "neg": ("m", lambda x: -x),
    "abs": ("m", abs), "tanh": ("m", math.tanh), "reciprocal": ("a", lambda x: 1.0 / x),
    "silu": ("m", lambda x: x * _sigmoid(x)),
    "gelu": ("m", lambda x: x * _sigmoid(1.702 * x)),   # fast approx (not erf)
}
REDUCE = {  # name -> list -> scalar
    "reduce_sum": sum, "reduce_max": max,
    "reduce_mean": lambda xs: sum(xs) / len(xs),
}
ELEMENTWISE = set(BINARY) | set(UNARY) | {"dequantize"}


def _softmax(xs):
    mx = max(xs)
    ex = [math.exp(x - mx) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def _rmsnorm(xs):  # weight = 1, eps = 1e-6 (host_kernel_lib.h)
    rms = 1.0 / math.sqrt(sum(x * x for x in xs) / len(xs) + 1e-6)
    return [x * rms for x in xs]


def _quantize(xs):  # scale = max|x|/127 (min 1e-10); q = clamp(round(x/scale), -128, 127)
    scale = max((abs(x) for x in xs), default=0.0) / 127.0
    if scale < 1e-10:
        scale = 1e-10
    q = [max(-128, min(127, int(round(x / scale)))) for x in xs]
    return scale, q


def _dequantize(xs):  # mirrors main.c: int32_src[i] = (int32)(a[i]*100); out = src*0.01
    return [float(int(x * 100.0)) * 0.01 for x in xs]


# ---------------------------------------------------------------------------
# C emission helpers
# ---------------------------------------------------------------------------

def _floats(vals):
    out = []
    for i in range(0, len(vals), 8):
        out.append("  " + ", ".join(f"{v:.6f}f" for v in vals[i:i + 8]) + ",")
    return "\n".join(out)


def _ints(vals):
    out = []
    for i in range(0, len(vals), 16):
        out.append("  " + ", ".join(str(v) for v in vals[i:i + 16]) + ",")
    return "\n".join(out)


def _emit_float_array(name, n, vals, const=True):
    q = "const " if const else ""
    print(f"static {q}float {name}[{n}] __attribute__((aligned(8))) = {{")
    print(_floats(vals))
    print("};")


def _emit_int8_array(name, n, vals):
    print(f"static const int8_t {name}[{n}] __attribute__((aligned(8))) = {{")
    print(_ints(vals))
    print("};")


def _emit_int_array(name, n, vals, ctype):
    print(f"static const {ctype} {name}[{n}] __attribute__((aligned(8))) = {{")
    print(_ints(vals))
    print("};")


# ---------------------------------------------------------------------------
# Integer inputs + golden for the int8/int16 precision sweep.
# Only integer-meaningful ops get int variants (see ara_sweep.h applicability).
# Ranges are per-op so the exact-integer golden never overflows the target width
# (add/sub: sum fits; mul: product fits; reduce_sum widens to int32).
# ---------------------------------------------------------------------------

INT_BINARY = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
              "mul": lambda a, b: a * b, "max": max, "min": min}
INT_UNARY = {"relu": lambda x: max(0, x), "neg": lambda x: -x, "abs": abs}
INT_REDUCE = {"reduce_sum": sum, "reduce_max": max}

# Symmetric input range [-r, r] per op, per width.
_I8_R = {"add": 50, "sub": 50, "mul": 8, "max": 100, "min": 100,
         "relu": 100, "neg": 100, "abs": 100, "reduce_sum": 100, "reduce_max": 100}
_I16_R = {"add": 10000, "sub": 10000, "mul": 150, "max": 20000, "min": 20000,
          "relu": 20000, "neg": 20000, "abs": 20000, "reduce_sum": 20000, "reduce_max": 20000}


def _int_arr(n, r, seed):
    rng = random.Random(seed)
    return [rng.randint(-r, r) for _ in range(n)]


def _emit_int_variants(kernel):
    """Emit op_a_i8/op_b_i8/golden_i8 (+ i16) for int-capable kernels."""
    for ctype, suffix, ranges, seed_a, seed_b in (
            ("int8_t", "i8", _I8_R, SEED + 10, SEED + 11),
            ("int16_t", "i16", _I16_R, SEED + 12, SEED + 13)):
        r = ranges[kernel]
        a = _int_arr(N_BIG, r, seed_a)
        _emit_int_array(f"op_a_{suffix}", N_BIG, a, ctype)
        if kernel in INT_BINARY:
            b = _int_arr(N_BIG, r, seed_b)
            _emit_int_array(f"op_b_{suffix}", N_BIG, b, ctype)
            g = [INT_BINARY[kernel](a[i], b[i]) for i in range(N_BIG)]
            _emit_int_array(f"golden_{suffix}", N_BIG, g, ctype)
        elif kernel in INT_UNARY:
            g = [INT_UNARY[kernel](x) for x in a]
            _emit_int_array(f"golden_{suffix}", N_BIG, g, ctype)
        elif kernel in INT_REDUCE:
            vals = [INT_REDUCE[kernel](a[:n]) for n in SIZES]  # int32 scalars per size
            print(f"static const int32_t golden_reduce_{suffix}[{len(SIZES)}] = {{ "
                  + ", ".join(str(v) for v in vals) + " };")


# ---------------------------------------------------------------------------
# Sweep header (one kernel): big inputs + that kernel's golden
# ---------------------------------------------------------------------------

def emit_sweep_header(kernel):
    a, b, m = _inputs(N_BIG)
    print(f"// Auto-generated by ara_lib.py {kernel} (seed={SEED})")
    print("#pragma once")
    print(f"#define OP_MAX_LEN {N_BIG}")
    print()
    _emit_float_array("op_a_big", N_BIG, a, const=False)
    _emit_float_array("op_b_big", N_BIG, b, const=False)
    _emit_float_array("op_mixed_big", N_BIG, m, const=False)
    print()

    if kernel in BINARY:
        g = [BINARY[kernel](a[i], b[i]) for i in range(N_BIG)]
        _emit_float_array("golden_big", N_BIG, g)
    elif kernel in UNARY:
        key, fn = UNARY[kernel]
        src = {"a": a, "b": b, "m": m}[key]
        _emit_float_array("golden_big", N_BIG, [fn(x) for x in src])
    elif kernel == "dequantize":
        _emit_float_array("golden_big", N_BIG, _dequantize(a))
    elif kernel in REDUCE:
        vals = [REDUCE[kernel](a[:n]) for n in SIZES]
        print(f"static const float golden_reduce[{len(SIZES)}] = {{ "
              + ", ".join(f"{v:.6f}f" for v in vals) + " };")
    elif kernel in ("softmax", "rmsnorm"):
        fn = _softmax if kernel == "softmax" else _rmsnorm
        src = m if kernel == "softmax" else a
        for si, n in enumerate(SIZES):
            _emit_float_array(f"golden_v{si}", n, fn(src[:n]))
        print(f"static const float* const golden_vec[{len(SIZES)}] = {{ "
              + ", ".join(f"golden_v{si}" for si in range(len(SIZES))) + " };")
    elif kernel == "quantize":
        scales = []
        for si, n in enumerate(SIZES):
            scale, q = _quantize(m[:n])
            scales.append(scale)
            _emit_int8_array(f"golden_q{si}", n, q)
        print(f"static const int8_t* const golden_q[{len(SIZES)}] = {{ "
              + ", ".join(f"golden_q{si}" for si in range(len(SIZES))) + " };")
        print(f"static const float golden_scale[{len(SIZES)}] = {{ "
              + ", ".join(f"{s:.9g}f" for s in scales) + " };")
    else:
        raise SystemExit(f"ara_lib.py: unknown kernel '{kernel}'")

    # int8/int16 inputs + golden (only for integer-meaningful ops; fp16 reuses
    # the fp32 arrays above, cast at runtime in the app).
    if kernel in INT_BINARY or kernel in INT_UNARY or kernel in INT_REDUCE:
        print()
        _emit_int_variants(kernel)


# ---------------------------------------------------------------------------
# ci_ara correctness header (small N): inputs + golden for the checked subset
# ---------------------------------------------------------------------------

# Kernels ci_ara checks (binary + the simple unary set + reductions).
_CI_UNARY = ["relu", "neg", "abs", "exp", "sigmoid", "tanh", "sqrt", "reciprocal", "silu"]
_CI_BINARY = ["add", "sub", "mul", "div", "max", "min"]


def emit_ci_header(n=32):
    a, b, m = _inputs(n)
    print(f"// Auto-generated by ara_lib.py (ci, seed={SEED})")
    print("#pragma once")
    print(f"#define OP_TEST_LEN {n}")
    print()
    for name, vals in [("op_a", a), ("op_b", b), ("op_mixed", m)]:
        print(f"static float {name}[{n}] = {{")
        print("  " + ", ".join(f"{v:.6f}f" for v in vals))
        print("};")
    for name in _CI_BINARY:
        g = [BINARY[name](a[i], b[i]) for i in range(n)]
        print(f"static float golden_{name}[{n}] = {{")
        print("  " + ", ".join(f"{v:.6f}f" for v in g))
        print("};")
    for name in _CI_UNARY:
        key, fn = UNARY[name]
        src = {"a": a, "b": b, "m": m}[key]
        g = [fn(x) for x in src]
        print(f"static float golden_{name}[{n}] = {{")
        print("  " + ", ".join(f"{v:.6f}f" for v in g))
        print("};")
    print(f"static float golden_reduce_sum = {sum(a):.6f}f;")
    print(f"static float golden_reduce_max = {max(a):.6f}f;")
    print(f"static float golden_reduce_mean = {sum(a) / n:.6f}f;")


def main():
    args = sys.argv[1:]
    if args == ["--ci"]:
        emit_ci_header(32)          # ci_ara correctness header (small N + golden)
    elif len(args) == 1 and not args[0].startswith("-"):
        emit_sweep_header(args[0])  # ara_<kernel> sweep header (big inputs + golden)
    else:
        raise SystemExit("usage: ara_lib.py <kernel> | --ci   "
                         "(emits an op_test_data.h on stdout)")


if __name__ == "__main__":
    main()
