# Conditional execution: the mini-compiler API

## What this is for

The hardware can skip tasks. A gating task writes CERF groups when it completes, and every
task carrying an inactive group is dropped: no dispatch, no operand traffic, no compute. That
is a branch mechanism, and three workloads want it.

|                  | gates | branches per gate | selection | branches grouped | reconvergence |
|------------------|-------|-------------------|-----------|------------------|---------------|
| Mixture of experts | 1   | one per expert    | top-k     | one group each   | weighted sum  |
| Early exit         | 1   | two, exclusive    | threshold | one shared group | pick the one that ran |
| Speculative decode | K, chained | one per position | threshold | one group each | commit in place |

Every one of them needs the same four facts stated: **who decides, how, what is guarded, and
where the branches come back together.** The API is those four facts and nothing else.

## What a skip actually does

This matters more than anything else on the page, and it is the opposite of what an earlier
version of this document claimed.

**A skipped task still signals its dependants.** `cond_exec_skip` drops the descriptor from
the READY queue only (`bingo_hw_manager_top.sv:1052-1057`). The CHECKOUT queue is pushed
either way, with the descriptor retagged `task_type = 2'b01` (:1149-1155), and it is the
checkout path that fires `dep_set`. The cycle model mirrors it
(`model/bingo_sim_chiplet.py:299-305`).

So skipping means "execute as a no-op", which is exactly gated-SSA merge semantics, and a
combine is an ordinary fan-in: **one ungated node, one unconditional edge per branch.** No
accumulator chain, no ordering imposed between branches that are independent by construction.

Two consequences worth stating plainly:

* The hazard on an edge out of a gated task is **not** a deadlock. It is a **potentially
  unguarded read**: a consumer outside the group runs whether or not the producer did, and
  reads whatever the producer last left in its buffer. Stale data, no fault, no message.
  `_validate_cerf_cross_group_edges` refuses that edge unless something guards the read.
* A skipped task's `dep_set` fires EARLIER than a real one's, because it needs no done-queue
  match. That can let a producer's set overtake the drain of an earlier consumer sharing a
  reused dep tag, which `bingo_validate_no_hang` does not model -- it reasons about the graph
  and assumes every task runs. The routing sweep in `bingo_sim_check` is what covers it.

## Vocabulary

Named after the two conventions that already apply, rather than inventing a third.

* **fork**, **branch**, **combine** - from the mixture-of-experts literature, where the three
  stages of the layer are router, dispatch and combine.
* **select** - the merge itself. In gated SSA this is the gamma function: a merge whose
  predicate is an ordinary data input, which is why such a graph needs no separate control
  flow graph.

`join` is deliberately NOT used: in this compiler it already means an unconditional fan-in,
the multi-column check that `enable_multi_col_check` refers to.

## The API

```python
fork = dfg.bingo_conditional_fork(
    gating_node,                 # the task whose completion decides
    select,                      # how it decides; the existing cond_dic dict, unchanged
)

branch = fork.branch(
    nodes,                       # the tasks guarded by this branch
    invert=False,                # run when the group is INACTIVE
    group=None,                  # None = its own CERF group; pass another branch to share
)

fork.combine(
    node,                        # where the branches reconverge
    inputs,                      # one value per branch; any subset may be absent at run time
    kind,                        # 'weighted_sum' | 'select' | 'sum'
    weights=None,                # required by 'weighted_sum'; the predicate as a data operand
)

fork.weights                     # float[branches], renormalised over the winners
fork.activation                  # uint8_t[branches], 1 for a branch the gate selected
```

`select` keeps the dict that edges carry today, so the four modes and their payloads are
unchanged: `{'mode':'top_k','k':K}`, `{'mode':'threshold','threshold':T}`,
`{'mode':'static','write_mask':M}`, and `{'mode':'custom','kernel_name':...}`.

`fork.weights` and `fork.activation` are two views of ONE L3 record the gating kernel writes.
They are allocated on first use, which has to be after the last `branch()` call, because
their length is the branch count; the compiler re-checks that and says so if a branch was
declared later.

### Why a fork object and not one flat call

Early exit has two branches sharing one combine. A per-region `combine=` argument would state
that fact twice with nothing checking the two agree. The fork owns the combine once, which is
also what makes the shared-group case expressible: `group=` takes another branch.

## The three workloads

### Mixture of experts - N branches, k win, weighted sum

```python
fork = dfg.bingo_conditional_fork(router, select={'mode': 'top_k', 'k': 2})
for e in range(E):
    fork.branch(expert_lane[e])                      # one CERF group per expert
fork.combine(combine_node, inputs=[y[e] for e in range(E)],
             kind='weighted_sum', weights=fork.weights)
```

Built, in `single_chip/workloads/moe4_4cluster`. The expert lane is written as an ordinary
chain of loads, GEMMs, a SwiGLU and a down projection. The branch structure appears once,
where it is a decision, instead of on every edge.

### Early exit - two exclusive branches, pick the one that ran

```python
fork = dfg.bingo_conditional_fork(confidence, select={'mode':'threshold','threshold':0.9})
taken   = fork.branch(exit_head)                     # confident: stop here
fork.branch(remaining_layers, group=taken, invert=True)   # otherwise: keep going
fork.combine(merge, inputs=[y_exit, y_deep], kind='select')
```

One group, two branches, one inverted. `kind='select'` because exactly one input exists at
run time and there is nothing to weight.

### Speculative decoding - chained gates

```python
prev = verify
for i in range(K):                                   # K bounded and small
    f = dfg.bingo_conditional_fork(prev, select={'mode':'threshold','threshold':0.5})
    f.branch([accept[i]] + commit[i])
    prev = accept[i]                                 # this gate is itself guarded
```

The cascade is the new structure: gate `i` is a guarded node of fork `i-1`, so position `i`
commits only if every earlier position did. No new API - a gating node is just a node - but
see the limits below.

## Where the weights come from

No device core has an FPU: every hart on `snax_split_cluster` is `rv32ima`, and only the host
CVA6 can divide floats. Renormalising the top-k probabilities over the winners
(`w_e = p_e / sum of active p_e`) is a divide, so it cannot happen on the device at all.

It happens in `__host_bingo_kernel_cerf_gating`, where the winner loop already is, and the
result is published as `float[E]` next to the 0/1 mask. The combine then GATHERS those
weights: it loads each winner's word and hands the **FP32 bits** to the SIMD's StreamMap scale
CSR (`simd_pass_map`'s `a_bits`) without ever interpreting them as a number. The mask says
which experts to read; the combine never re-derives top-k, so there is exactly one copy of
that decision.

Renormalisation belongs to SELECTION, not to the softmax and not to the accumulation: the raw
softmax sums to one over ALL branches, so using it unrenormalised scales the output down by
the mass of the branches that did not run. That is a silent wrong answer, which is the worst
kind.

## What the compiler does

1. Mark the gating node `task_type = gating`, allocate a CERF group per branch (or reuse the
   shared one), set `cond_exec_en` / `cond_exec_group_id` / `cond_exec_invert` on the guarded
   nodes.
2. Build the gating kernel args from `select`, and allocate the one L3 selection record.
3. Add any missing `input -> combine` edge, so the workload states the fan-in once, and bind
   the mask and the weights into the combine's kernel args through `bind_combine()`.
4. Allow the declared combine through `_validate_cerf_cross_group_edges`, which otherwise
   refuses a cross-group edge as an unguarded read.

## What the compiler refuses

The value of declaring the construct is that these become build errors instead of silence.
Each has a test in `mini_compiler/test_conditional_fork.py`.

1. A branch that does not reach the combine.
2. `kind='weighted_sum'` without `weights`.
3. Two branches sharing a group with neither inverted, feeding a `select` - more than one
   input can exist, so the merge is ambiguous.
4. A `select` policy that disagrees with a `cond_dic` on an edge from the same gate. This
   used to be silent: the compiler took whichever it processed first.
5. A conditional target that no branch declares, and a second combine on one fork.
6. Any unconditional edge out of a guarded node to a node outside its group that is not the
   declared combine, does not carry the SW guard, and is not compiler-inserted plumbing.
7. A declared combine that closes only some of its fork's branches.

## Compatibility

`cond_dic` on edges keeps working and lowers to the same thing; the fork is a second spelling,
not a replacement. Only `moe4_4cluster` ever used the edge form and
`bingo_define_conditional_region` has no callers, so backwards compatibility was never a real
constraint here.

## Known gaps

* **Cascaded gating is unverified.** The packed descriptor allows it: `cond_exec_en` and
  `task_type` are independent fields, so a task can be both gated and gating. The compiler's
  auto-insert path has never produced one. Needs a mechanism test of its own before anything
  is built on it.
* **`invert` has no silicon coverage.** It is plumbed through to `cond_exec_invert` and the
  fork reaches it, but no workload has run one.
* **CERF-gated HOST tasks deadlock.** Four gated host store/check pairs on cluster 0's host
  core hang the cycle model in every routing scenario but "all branches active", through the
  early-dep_set race described above. `moe4_4cluster` therefore keeps its per-expert
  verification ungated, by zeroing the combine's landing area first so a loser's slot is
  checkable as zeros. Worth root-causing in the tag allocator before anything relies on
  gating a host task.
* **There is no loop.** CERF decides once per dispatch. Speculative decoding runs a
  data-dependent number of ROUNDS, which this cannot express - in gated-SSA terms there is a
  gamma but no mu or eta. Either bound the rounds and unroll, or let the host re-dispatch per
  round and pay a round trip.
