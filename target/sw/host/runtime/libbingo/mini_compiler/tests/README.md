# tests/ — runnable checks

No framework. Each file is a script that prints one line per check and exits non-zero if
any failed:

```
pixi run python3 tests/test_libs.py
```

(`pixi run` from `snax_cluster`, because the host python is 3.9 and this needs 3.10+.)

| file | what it covers |
|---|---|
| `test_libs.py` | The block/port contract: what a mismatch is refused for, what the linker joins, whether a producer's buffer becomes reusable by the consumer, and the namespacing that lets one block be instantiated twice. |
| `test_static_l1.py` | The liveness rule and the packer: which buffers may share, and that the placement the packer returns is provably safe. |
| `test_conditional_fork.py` | CERF: the gating masks, and that every routing scenario terminates. |

## What these do not cover

They check the compiler, not the hardware. A graph that passes here can still be wrong on
silicon — the sim check in `passes/bingo_sim_check.py` is a deadlock oracle, not a model of
what the kernels compute. The real gate for a change to `libs/` is that a workload's
generated header is byte-identical before and after, which is cheap to run and catches
anything that moved the schedule.
