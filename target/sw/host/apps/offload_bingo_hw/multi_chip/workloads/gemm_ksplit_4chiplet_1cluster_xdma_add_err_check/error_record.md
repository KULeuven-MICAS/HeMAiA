# Error Record: gemm_ksplit_4chiplet_1cluster_xdma_add_err_check

## Purpose

This workload preserves the original failing `gemm_ksplit_4chiplet_1cluster_xdma_add`
DFG shape as a regression test. It should expose the ordering bug where a host
check can run before the host IDMA task has refreshed the golden-data scratch
buffer.

## Workload Shape

- Four chiplets, one cluster per chiplet.
- `K` is split into four chunks, so each chiplet computes one partial GEMM
  result `D_partial_k{i}`.
- Each chiplet stores its partial result to local L3, loads its local golden
  partial result to local L3 scratch, and checks the partial result.
- Chiplet 0 then pulls remote partial results from the other chiplets' cluster-0
  L1 memories into chiplet-0 L1.
- Chiplet 0 performs xDMA int32 add reduction in L1:
  - `k0 + k1 -> sum_k0_to_k1`
  - `sum_k0_to_k1 + k2 -> sum_k0_to_k2`
  - `sum_k0_to_k2 + k3 -> sum_k0_to_k3`
- After each add, the workload loads the corresponding golden running sum into
  `l3_check_scratch[0]` and immediately checks the L1 `dst_sum` buffer.
- The final running sum is then dequantized in place and checked against the
  FP32 golden output.

## Observed Failure

The partial GEMM checks pass, but the first running-sum check fails. The UART
log shows:

```text
[sum_k0_to_k1] output[0]=0, golden[0]=35
[sum_k0_to_k1] output[1]=33, golden[1]=7
[sum_k0_to_k1] output[4]=52, golden[4]=244
[sum_k0_to_k1] output[5]=13, golden[5]=254
```

The standalone `xdma_int32_add_4chiplet_1cluster` workload passes, so this is
not expected to be an xDMA add arithmetic problem.

## Suspected Root Cause

The suspicious sequence is:

```text
XDMA_Add_k0_to_k1
  -> Load_Golden_sum_k0_to_k1
  -> Check_sum_k0_to_k1
```

`Load_Golden_sum_k0_to_k1` writes the golden running sum into
`l3_check_scratch[0]`. `Check_sum_k0_to_k1` then reads the same scratch buffer.
The DFG has an explicit dependency from the load to the check, but the generated
task tags/dependency checks appear not to preserve this ordering reliably for
host IDMA followed by host check on the same chiplet.

As a result, the check may read stale contents from `l3_check_scratch[0]`, such
as the previous `D_partial_k0` golden data, before the golden running sum has
been copied in.

## Expected Framework Fix

Fix the Bingo DFG/tag generation so this ordering is preserved:

```text
device node -> host IDMA load -> host check
```

on the same chiplet. After the framework fix, this `*_err_check` workload should
pass without changing the workload flow.

The fixed non-error workload avoids this specific failure mode by completing all
xDMA add reductions first, storing the final sum back to L3, loading the final
golden sum to L3, checking once, and only then dequantizing and checking FP32.
