# passes/ — what runs over an assembled graph

Each pass is a mixin on `BingoDFG`, and `bingo_compile_dfg` runs them in a fixed order
because each depends on the last. The analyses the passes use live here too.

## The passes, in order

| file | what it does | what it catches |
|---|---|---|
| `bingo_dfg_transforms.py` | Entry node, per-core exit chain, core-sequencing edges, dummy set/check nodes. Runs in two halves, with the conditional pass between them. | — |
| `bingo_dfg_conditional.py` | Wires the CERF conditional-execution registers: which node gates which, and what a skipped task still signals. | A skipped task IS pushed to the checkout queue retagged, so it still fires `dep_set` — a consumer that assumed otherwise waits forever. |
| `bingo_dfg_descriptor.py` | Assigns dep set/check info, allocates tags by a minimum chain cover, packs each node into a task descriptor. | Tag exhaustion, and descriptor-width overflow. |
| `bingo_dfg_validate.py` | `bingo_validate_no_hang` after the tags are allocated; placement and handle validation from inside the emit pass. | A GEMM node on the wrong hart programs *that* hart's accelerator at the same CSR offsets and reports success. Nothing faults. |
| `bingo_dfg_staticl1.py` | Places cluster buffers, reusing memory between buffers whose lifetimes cannot overlap. | Overflowing the 514,816 B heap — now a build error, since `l1_capacity_bytes` is set. |
| `bingo_dfg_emit.py` | Writes the C, validating placement and handles first. | Two same-named handles emitting two declarations that do not compile. |
| `bingo_dfg_report.py` | The CSV, the graph PNGs, the L1 occupancy plot, and the per-block pictures (below). | — |

## The analyses

| file | |
|---|---|
| `bingo_liveness.py` | Who uses which buffer, reachability between nodes, and `can_share` — the rule static-L1 rests on: every user of A must be a graph ancestor of every user of B. One incomparable pair vetoes the whole buffer pair. |
| `bingo_l1_packer.py` | The placement search itself, and the proof that the placement is safe. |
| `bingo_sim_check.py` | Replays the graph against a model of the hardware task manager. It is a **deadlock oracle**, not a performance model — trust it for "this wedges", not for cycle counts. |
| `bingo_block_viz.py` | One picture and one text listing per libs block, under `<output_dir>/block_dfg/`, drawn from the records `Pipeline.run()` leaves on the DFG. It runs first in `bingo_compile_dfg`, so it shows the dependencies the blocks stated rather than the dummy nodes they are lowered to, and it never fails a build. `BINGO_BLOCK_DFG=0` skips it. |

## Why the order is fixed

Dependency tags are allocated over the graph *after* the exit chain, the conditional
regions and the sequencing edges exist, so running the descriptor pass earlier would cover
a different graph. Static L1 runs after that because liveness is computed over the final
edge set. Emission is last because it reads the results of all of them — which is also why
the placement check lives there rather than up front: it runs at the point where a wrong
answer would actually be written out.

Note that **validation is not a single pass**. `bingo_dfg_validate.py` contributes the hang
check (after tag allocation), the two CERF checks (right after the conditional pass), and
the placement/handle checks (called from emit). Reading it as "step 1" is wrong.
