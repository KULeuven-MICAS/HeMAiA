#!/usr/bin/env python3
# Test workload for __host_bingo_kernel_int32_add (the new kernel used
# for inter-cluster partial-D accumulation in K-split GEMM schemes).
#
# Generates 4 random INT32 partial arrays (simulating cluster outputs)
# and the expected sum. The DFG chains 3 add calls to reduce 4 inputs.

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402

np.random.seed(42)


def emit_header_file(**kwargs):
    emit_str = ["#include <stdint.h>"]
    emit_str += emit_int32_add_data(**kwargs)
    return "\n\n".join(emit_str)


def emit_int32_add_data(**kwargs):
    data_str = []
    num_elements = kwargs["num_elements"]
    data_str.append(format_scalar_definition("uint32_t", "num_elements", num_elements))

    # 4 input partial-D arrays (simulate 4 clusters contributing INT32 partials)
    partial_0 = np.random.randint(-1000000, 1000000, size=num_elements, dtype=np.int32)
    partial_1 = np.random.randint(-1000000, 1000000, size=num_elements, dtype=np.int32)
    partial_2 = np.random.randint(-1000000, 1000000, size=num_elements, dtype=np.int32)
    partial_3 = np.random.randint(-1000000, 1000000, size=num_elements, dtype=np.int32)

    # Expected golden: running sum
    golden_01 = (partial_0.astype(np.int64) + partial_1.astype(np.int64)).astype(np.int32)
    golden_012 = (golden_01.astype(np.int64) + partial_2.astype(np.int64)).astype(np.int32)
    golden_final = (golden_012.astype(np.int64) + partial_3.astype(np.int64)).astype(np.int32)

    data_str += [
        format_vector_definition("int32_t", "golden_partial_0", partial_0),
        format_vector_definition("int32_t", "golden_partial_1", partial_1),
        format_vector_definition("int32_t", "golden_partial_2", partial_2),
        format_vector_definition("int32_t", "golden_partial_3", partial_3),
        format_vector_definition("int32_t", "golden_sum_01", golden_01),
        format_vector_definition("int32_t", "golden_sum_012", golden_012),
        format_vector_definition("int32_t", "golden_sum_final", golden_final),
    ]
    data_str.append(format_scalar_definition("uint32_t", "array_bytes", num_elements * 4))
    return data_str


def main():
    parser = argparse.ArgumentParser(description="int32_add test data")
    parser.add_argument("-c", "--cfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with open(args.cfg) as f: param = hjson.loads(f.read())
    with open(args.hwcfg) as f: hw = hjson.loads(f.read())
    merged = {**param, **hw}
    content = emit_header_file(**merged)
    with open(args.output, "w") as f: f.write(content)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
