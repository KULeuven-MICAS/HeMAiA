# passes/ — what runs over an assembled graph

Each pass is a mixin on `BingoDFG`, and `bingo_compile_dfg` runs them in a fixed order
because each depends on the last. The analyses the passes use live here too.

## The passes, in order

| file | what it does | what it catches |
|---|---|---|
| `bingo_dfg_hazards.py` | Every two tasks of which one writes bytes the other touches are ordered by the graph's dependency and ordering-only edges, from what each kernel declares in `accesses()`; a dispatch that reads until its wait (`operands_held_by`) counts to the wait. Enforced where the compiler orders the cores, reported on otherwise. | A pair left unordered (named with the edge to add), a kernel without `accesses()`, an address field left out of it. |
| `bingo_dfg_schedule.py` | Each device core's program, chosen by list scheduling over the cost model (`dfg.schedule_cores`), each cluster's ring loads in the ring's order; laid out with and without a one-task lookahead (a core waits for a task of a longer path that lands while the ready one would hold it), the shorter kept. Where its order overflows the static L1 plan, it schedules again with an L1 budget: a task about to open a buffer gets an edge from the last users of buffers that are done with on its cluster, so the packer may give it their space (or waits); only where that does not fit either, each core's tasks in the order they were made (a task that waits is not passed by the core's later ones, unless that order closes a cycle). With `dfg.async_exec`, every task whose kernel declares a `Split` (VersaCore GEMM and FlashAttention matmuls, the SIMD's staged and softmax kernels, the xDMA's 6D transfer and ChainGather) may run as configure / start / wait parts the schedule places under its accelerator's rules (one pending configuration, the works in flight, an exclusive kernel); parts left side by side are fused again, and the split programs are kept only where the model says they finish sooner. Then the counter-tracked works' numbers, the shared setups (VersaCore's shape table), and the CSR reuse along each accelerator core's program (`config_key` / `config_reuse`; a configure part that would only write addresses folds into its start; the softmax geometry, the ChainGather clean-up). | A task that never becomes ready under its accelerator's rules; a `gemm_wait` the compiler did not make. |
| `bingo_dfg_transforms.py` | Entry node, per-core exit chain, core-sequencing edges, fan-out pruning (on by default), the tag budget, dummy set/check nodes, then dep set/check info and the tag allocation (a minimum chain cover). Runs in two halves, with the conditional pass between them. | A cell that needs more tags than `2**dep_tag_width` is fitted by added ordering edges, or by relay tasks where ordering cannot fit it; only a cell neither can fit raises. Tag exhaustion in the allocator raises too. |
| `bingo_dfg_conditional.py` | Wires the CERF conditional-execution registers: which node gates which, and what a skipped task still signals. | A skipped task IS pushed to the checkout queue retagged, so it still fires `dep_set` — a consumer that assumed otherwise waits forever. |
| `bingo_dfg_descriptor.py` | The descriptor field table (LSB to MSB, mirrors `bingo_hw_manager_task_desc_t`), packing, unpacking and the task-list word split. | Descriptor-width overflow, and a field value that does not fit its width. |
| `bingo_dfg_validate.py` | `bingo_validate_no_hang` after the tags are allocated; placement and handle validation from inside the emit pass. | A GEMM node on the wrong hart programs *that* hart's accelerator at the same CSR offsets and reports success. Nothing faults. |
| `bingo_dfg_staticl1.py` | Places cluster buffers, reusing memory between buffers whose lifetimes cannot overlap. Writes the L1 layout CSV and plot. | Overflowing the cluster's L1 heap, when the workload sets `l1_capacity_bytes` and passes `static_l1`. |
| `bingo_dfg_emit.py` | Writes the C, validating placement and handles first. | Two same-named handles emitting two declarations that do not compile. |
| `bingo_dfg_report.py` | The CSV, the graph PNGs, and the per-block pictures (below). | — |

## The analyses

| file | |
|---|---|
| `bingo_cost_model.py` | How long a task holds its core, a dispatch's array time and an edge's latency, in cluster cycles: formulas over the task's arguments and a few constants, the same for every graph, set from the SNAX cluster replays of the LLM stage apps (each kernel's fixed and per-element cost; a configure part's by kernel, `configure_cycles`). What `bingo_dfg_schedule.py` orders the cores by and decides the splits with. |
| `bingo_liveness.py` | Who uses which buffer, reachability between nodes, and `can_share` — the rule static-L1 rests on: every user of A must be a graph ancestor of every user of B. One incomparable pair vetoes the whole buffer pair. With the reach labels' column form, a buffer's users are summarised once and compared a whole row at a time (`build_interference`). |
| `bingo_reach.py` | Every node's descendants in the happens-before graph at once, one bitset per node over a topological order: the reachability the tag allocator and the hang check test thousands of times. |
| `bingo_l1_packer.py` | The placement search itself, the proof that the placement is safe, and the layout CSV and plot writers. |
| `bingo_exec_export.py` | Writes the compiled graph (stream order, dummies, args, ring tables) as Execution-IR JSON for the bingo framework's DSE (`bingo_export_exec_ir`, `ring_table`). |
| `bingo_sim_check.py` | Replays the graph against a model of the hardware task manager. It is a **deadlock oracle**, not a performance model — trust it for "this wedges", not for cycle counts. |
| `bingo_block_viz.py` | One picture and one text listing per libs block, under `<output_dir>/block_dfg/`, drawn from the records `Pipeline.run()` leaves on the DFG. It runs first in `bingo_compile_dfg`, so it shows the dependencies the blocks stated rather than the dummy nodes they are lowered to, and it never fails a build. `BINGO_BLOCK_DFG=0` skips it. |

## Rewrites a workload calls itself

| file | |
|---|---|
| `bingo_args_phases.py` | `bingo_args_phases(dfg, budget, dev_cores, dm, node)`: cuts each cluster's device tasks, in run order, into args phases of at most `budget` bytes (params `args_phase_bytes`) and adds each later phase's load, an iDMA copy on the DM core, with its edges: after the last task of phase p - 2 on every core (the L1 region it overwrites), before every core's first task of phase p. `bingo_dfg_emit.py` (ARGS PHASES) gives the even and odd phases the two alternating L1 regions. `bingo_compile_dfg` runs it once the schedule has fixed the programs, when `dfg.args_phases` holds its arguments (`ChipletBuild.finish` sets them from params `args_phase_bytes`); `_args_bytes(t)` is a task's args struct in the arena. |

## Why the order is fixed

Dependency tags are allocated over the graph *after* the exit chain, the conditional
regions and the sequencing edges exist, so allocating tags earlier would cover
a different graph. The tag budget runs between: after pruning, so it does not order edges
the pruning would have dropped, and before the dummy passes, so the edges it adds are
lowered like the rest. It sees each cell as it will be after lowering, with an order no
stronger than the allocator's, so a cell it fits is one the allocator can tag. Static L1 runs after that because liveness is computed over the final
edge set. Emission is last because it reads the results of all of them — which is also why
the placement check lives there rather than up front: it runs at the point where a wrong
answer would actually be written out.

Note that **validation is not a single pass**. `bingo_dfg_validate.py` contributes the hang
check (after tag allocation), the two CERF checks (right after the conditional pass), and
the placement/handle checks (called from emit). Reading it as "step 1" is wrong.
