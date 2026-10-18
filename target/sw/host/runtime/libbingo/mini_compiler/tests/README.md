# tests/ — runnable checks

No framework, and pytest collects nothing from them. Each file is a script that prints one
line per check and exits non-zero if any failed. Run them one at a time with a Python 3.10+
that has `networkx`, `numpy` and `hjson` (the host's default python3 is 3.9):

```
python3 tests/test_libs.py
```

The SIMD beat is a property of the cluster cfg and is not read from a file here: each script
sets it with `set_simd_beat_bytes` before building a SIMD task.

| file | what it covers |
|---|---|
| `test_libs.py` | The block/port contract: what a mismatch is refused for, what the linker joins, whether a producer's buffer becomes reusable by the consumer, and the namespacing that lets one block be instantiated twice. |
| `test_static_l1.py` | The liveness rule and the packer: which buffers may share, and that the placement the packer returns is provably safe. |
| `test_conditional_fork.py` | The conditional fork/branch/combine API: CERF group assignment, invert, the refusals, and the one-group-per-core rule. |
| `test_tag_budget.py` | The dependency-tag budget: a gather that needs 16 tags lowers and tags at 4 once the pass has added ordering edges, a graph that fits is left alone, a fan-in no ordering can fit is fitted through a relay, and broadcast dep-sets are counted, and split, per cell. |
| `test_gemm_async.py` | The queued GEMM dispatch: the start kernels' descriptors, `GemmQueue` numbering and start/wait pairs, the core program order, and the `fa_sweep` built on them. |

## What these do not cover

They check the compiler, not the hardware. A graph that passes here can still be wrong on
silicon — the sim check in `passes/bingo_sim_check.py` is a deadlock oracle, not a model of
what the kernels compute. The real gate for a change to `libs/` is that a workload's
generated header is byte-identical before and after, which is cheap to run and catches
anything that moved the schedule.
