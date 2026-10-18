# passes/ — what runs over an assembled graph

Each pass is a mixin on `BingoDFG`, and `bingo_compile_dfg` runs them in a fixed order
because each depends on the last. The analyses the passes use live here too.

## The passes, in order

| file | what it does | what it catches |
|---|---|---|
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
| `bingo_liveness.py` | Who uses which buffer, reachability between nodes, and `can_share` — the rule static-L1 rests on: every user of A must be a graph ancestor of every user of B. One incomparable pair vetoes the whole buffer pair. |
| `bingo_reach.py` | Every node's descendants in the happens-before graph at once, one bitset per node over a topological order: the reachability the tag allocator and the hang check test thousands of times. |
| `bingo_l1_packer.py` | The placement search itself, the proof that the placement is safe, and the layout CSV and plot writers. |
| `bingo_exec_export.py` | Writes the compiled graph (stream order, dummies, args, ring tables) as Execution-IR JSON for the bingo framework's DSE (`bingo_export_exec_ir`, `ring_table`). |
| `bingo_sim_check.py` | Replays the graph against a model of the hardware task manager. It is a **deadlock oracle**, not a performance model — trust it for "this wedges", not for cycle counts. |
| `bingo_block_viz.py` | One picture and one text listing per libs block, under `<output_dir>/block_dfg/`, drawn from the records `Pipeline.run()` leaves on the DFG. It runs first in `bingo_compile_dfg`, so it shows the dependencies the blocks stated rather than the dummy nodes they are lowered to, and it never fails a build. `BINGO_BLOCK_DFG=0` skips it. |

## A rewrite a workload calls itself

| file | |
|---|---|
| `bingo_gemm_reuse.py` | `reuse_gemm_config(dfg)`: a `gemm_full` that repeats the previous `gemm_full`'s configuration on its core (every field but the four operand addresses) becomes `gemm_minimal`. It reads creation order as run order, so the cores must have been ordered first (`GemmQueue.program_order`). `bingo_compile_dfg` does not run it. |

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
