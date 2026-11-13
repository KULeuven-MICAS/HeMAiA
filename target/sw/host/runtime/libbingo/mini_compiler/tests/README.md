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
| `test_collectives.py` | The compiler's order of the broadcasts and gathers (`passes/bingo_dfg_collectives.py`, `libs/blocks/collective.py`): two broadcasts ordered unless one push-engine queue runs both, two gathers into overlapping bytes ordered, a broadcast beside a gather and gathers into disjoint bytes left alone, a receiver's wait counting as a broadcast done, an ordering-only edge on one core and a dependency across cores, and the report-only mode. |
| `test_conditional_fork.py` | The conditional fork/branch/combine API: CERF group assignment, invert, the refusals, and the one-group-per-core rule. |
| `test_tag_budget.py` | The dependency-tag budget: a gather that needs 16 tags lowers and tags at 4 once the pass has added ordering edges, a graph that fits is left alone, a fan-in no ordering can fit is fitted through a relay, and broadcast dep-sets are counted, and split, per cell. |
| `test_schedule.py` | The compiler's core programs (`passes/bingo_dfg_schedule.py`): the list schedule, ring loads behind their ring's post, and with `async_exec` the configure / start / wait splits of every accelerator (numbering, depth, exclusive kernels, the shape table, the CSR reuse and the fold of an address-only configure into its start), the edges that let the static L1 plan reuse space where the schedule overflows it, and the made order of the fallback (no passing a task that waits, except to break a cycle). |
| `test_hazards.py` | The memory-hazard check (`passes/bingo_dfg_hazards.py`): unordered overlapping accesses refused with the edge to add, the declarations (`accesses()`, strided and nested runs), and every kernel class's declaration. |
| `test_l1_heap.py` | A cluster's L1 heap as the compiler counts it (`passes/bingo_l1_heap.py`): every args struct's size against the C compiler's `sizeof` (where a C compiler is on the path), the heap's fragments and size, the list costs, and the args-phase budgets and static capacity of a cluster that fits and of one that does not. |
| `test_ring_push.py` | The weight rings (`libs/blocks/load_stream.py` `WeightRings.push_from`): one post a run on the ring's DM core, and the edges that are the protocol: the loads of the bytes a run overwrites before its push, the pushes in ring order, the queue share, each load after its run's push; a ring both clusters read (`post_chunk` / `read_chunk`), every post after the last read of the bytes it overwrites, that read made before or after the post; and another poster between a ring's runs (`boundary`, `post_after`, `tail_before`): the run boundaries, the edges around the other poster, `runs_of`; a load or post before the push engine is set. |
| `test_xdma_chain_split.py` | A ChainGather's parts as the compiler makes them: the crossing check between two collectors and the CSR clean-up rule. |

## What these do not cover

They check the compiler, not the hardware. A graph that passes here can still be wrong on
silicon — the sim check in `passes/bingo_sim_check.py` is a deadlock oracle, not a model of
what the kernels compute. The real gate for a change to `libs/` is that a workload's
generated header is byte-identical before and after, which is cheap to run and catches
anything that moved the schedule.
