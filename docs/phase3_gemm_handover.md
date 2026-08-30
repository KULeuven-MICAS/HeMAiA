# One GEMM split across 64 workers — handover

Bora Barutçu, MICAS internship, 6 July – 8 September 2026.
Supervisors: Yunhao Deng, Xiaoling Yi, Fanchen Kong.

This is meant to be read first, in about fifteen minutes, by someone who has to
pick the work up. It says what was built, what is actually proven, what only
looks proven, and what I would do next.

---

## 1. What was built

**Goal:** take one GEMM and split it across M, K and N over 16 chiplets × 4
clusters = 64 workers, using the Bingo HW dependency manager for scheduling.

**It works.** The largest run is a real 4 × 8192 × 4096 GEMM on all 64 workers:
PASS in 7h14m, 16/16 checks correct with **full coverage** (the checks read every
byte of every worker's output slice, not a sampled window), 128/128 core exits
clean, zero AXI width warnings. At that shape the partitioned operand image is
*required*, not merely exercised: replicated, the operands would need 32 MiB on a
16 MiB memory chiplet.

Four things had to be built to get there, each validated on its own RTL run with
a control that changed exactly one thing:

| what | why it was needed |
|---|---|
| worker grid: `(w_m, w_n)` → chiplet, `w_k` → cluster | puts every partial of one output cell on the same chiplet, so each K-reduction is chip-local and no reduction crosses D2D |
| topology-independent D2D link init | chiplets at a coordinate ≥ 2 used to hang; a 3-chip row whose third chip was silent now passes 3/3 |
| several memory chiplets, with the operand image **partitioned** across them (A replicated, B split by N) | 4 × 16 MiB replicated is one chip's usable capacity; partitioned it is ~92% of a single 64 MiB chip, and keeps the bandwidth |
| generated-code size: host task tables as rodata, per-workload device kernel table | device binary 87392 → 13500 B, generated host code 585368 → 267096 B |

---

## 2. Running it

The nine validated configurations are CI tasks, so reproducing a result does not
require knowing anything written here:

```sh
python3 target/sim/automation/ci/local_ci/scalability_exp_3chip_1c_row/run_local_ci.py
```

Suites, cheapest first:

| suite | what it covers | simulation time |
|---|---|---|
| `local_ci/scalability_exp_3chip_1c_row/` | 3 chiplets in a row — the cheapest multi-chip test | 1m12s |
| `local_ci/tapeout_1c/`, `local_ci/tapeout_2c/` | the M, M+N, M+K+N, tile-pack and accumPrevC rungs (4 and 8 workers) | minutes each |
| `local_ci/scalability_exp_4chip_4c/` | 16 workers, `grid_K = 4` | 1h32m |
| `local_ci/scalability_exp_16chip_4c_4mem/` | 64 workers, 16 chiplets × 4 clusters | 29m |
| `.../scalability_exp_16chip_4c_4mem/task_long_thinM.yaml` | the shape that needs the partitioned image. **Needs `HEMAIA_SIM_TIMEOUT_H=8`** — the default cap is 4 h and it will end the run with a bare `FAIL` | 7h14m |

The mechanism is an optional `HOST_DATA_CFG` key in the task list, holding a
repo-root-relative path to one of `.../gemm_parametric/params/`. Without it the
workload builds its default `params.hjson`. It is deliberately **not** called
`DATA_CFG`: that name is a repo-wide convention used by ~69 host workloads and by
every device application's `data/Makefile`, so a command-line `DATA_CFG=` would
propagate through recursive make and silently replace the device app's data
config with the host's.

**Size a run before launching it.** Simulation runs at roughly **90 µs of
simulated time per wall-clock minute** once the workers are active, and that rate
barely depends on chiplet count — a 17-chiplet and a 5-chiplet platform agree to
4%. Whole-run averages look 2–5× better than this because they are diluted by the
cheap host-only boot phase; size on the compute-phase rate.

Use `--engine vcs`: it is **7.8× faster than vsim** on identical work.

---

## 3. Where things are

```
target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/gemm_parametric/
    gemm_parametric.py   builds the DFG; worker grid, tiling, memory-chiplet partition
    gemm_check_plan.py   tile choice + check plan. SHARED with the datagen ON PURPOSE, see §5
    gemm_datagen.py      golden model, operand images, per-memory-chiplet images
    params.hjson         the default configuration
    params/              the nine validated ones, with a README pairing each to its RTL cfg

target/sw/shared/runtime/hemaia_d2d_link.h                    hemaia_d2d_link_initialize()
target/sw/host/runtime/libbingo/mini_compiler/bingo_dfg.py    the mini-compiler
target/sw/device/toolchain.mk, apps/base.template.ld          device code-size fix
target/rtl/cfg/hemaia_scalability_exp_*.hjson                 the platforms these runs need
```

---

## 4. What is proven, and what only looks proven

This is the section I would most want to read if I were picking this up.

**Proven in RTL:** the 64-worker M+K+N split at a size that needs the partitioned
image; the D2D topology fix (and chiplets that worked before still work); both
code-size fixes — for the device binary the whole 100-application software matrix
is identical to the control, which is what rules out having dropped a kernel that
something needed.

**Verified offline but never exercised in RTL:**

- The iDMA leg of the trace sink. It should work; nothing has run with it.
- The `task_id` overflow guard. `task_id` is 12 bits and Python integers do not
  truncate, so a node id ≥ 4096 used to bleed into `assigned_chiplet_id` and
  dispatch the task to the wrong chiplet — it links, runs, and returns a wrong
  answer. The guard is unit-tested, but no configuration large enough to trip it
  has been run.
- The merged memory-chiplet loader in `load_binary.sv.tpl`. Every partitioned run
  used the older form of that file; the merge with the current upstream rewrite
  has not been simulated.

**NOT established, despite appearances:**

- **The bandwidth benefit of several memory chiplets is unmeasured.** Runs with 1
  and with 4 memory chiplets sit next to each other with similar wall-clock
  times, and that comparison means nothing: the matrix was tiny, and machine load
  moves wall-clock by ±60%. Measuring it means one run of the thin-M
  configuration against a single 64 MiB chip, about 7 hours.
- **The wall-clock model is not trustworthy.** It predicted 2.4 h for the run that
  took 7.14 h. The compute-phase *rate* (~90 µs/min) is solid; the model that
  turns a matrix shape into a task count into a wall-clock is not.
- **Accelerator utilisation is not a solved number.** Per-core occupancy came out
  at 81.6% with 95.8% of kernel time inside `SIMD_RUN`, while the arithmetic
  roofline for the same run is 0.19%. The gap is real and unexplained: the time
  is inside the compute kernel, not in DMA or argument handling. Settling it
  needs finer per-kernel events, not more runs. **Do not quote 0.19% as an
  efficiency figure** — every shape that has completed here was chosen to prove
  something structural cheaply, and is memory-bound by construction.
- **Every large run used a 16 MiB `spm_wide`**, so none of them proves anything
  about fitting the 128 KiB the tapeout actually has. That is what makes the
  code-size work a genuine structural constraint rather than a simulation
  artefact.

---

## 5. Traps that each cost a day

- **B is N-major, not K-major.** The operand's numpy shape and its semantic
  blocking are different things: `block_gemm_golden_model` reshapes the same flat
  bytes to `(N1, K1, meshCol, tileSize)`. Read those reshapes before computing any
  offset. Invisible until `grid_N > 1`, because the two formulas coincide at
  `N1 = 1`.
- **`gemm_check_plan.py` is shared between the DFG builder and the datagen
  deliberately.** Both must choose the identical tile shape, because the result
  buffer's layout is derived from it. If they ever diverge the checker compares
  the wrong bytes and cheerfully validates a wrong answer. `mem_chip_n_ranges()`
  is the same kind of single source of truth for the partitioned image.
- **`accumPrevC` requires a 1×1 inner tile.** The accumulator is a single
  VersaCore register holding one tile, so with a larger tile it silently produces
  wrong results — sign flips on every chip, no error of any kind.
- **Do not regression-check by diffing the generated header.** Dependency-tag
  *numbers* shift when unrelated allocation order changes, so a byte diff fails
  for reasons that mean nothing. Compare decoded fields.
- **`router_to_soc_iw` can only ever equal `soc_to_router_iw`.** `hemaia_d2d_link`
  has one `axi_req_t` serving both its input and its output ports, so setting it
  higher misaligns the struct rather than widening an ID. It used to build fine
  and give a plausible wrong answer hours later; port width mismatches are now
  errors (`-error=PCWM-L`, `-error 2241`), so it fails at elaboration instead.
  The comment `// soc_to_router_iw + clog2(chips)` in the older cfgs is not what
  the hardware obeys.
- **Trace bytes are not progress, and a frozen host trace is not a hang.** A host
  that has dispatched its work parks and retires nothing. A host blocked on the
  ready-queue read at `0x0a001000` is idle *by design*; a host stopped on the
  store to `0x0b00001c` in `bingo_hw_scheduler_init` really is hung.
- **Adding a memory chiplet to a cfg is two edits**: the coordinate/size entry and
  `bender_target: ["hemaia_mem_chip"]` beside it. Without the second, elaboration
  fails with `Module 'hemaia_mem_chip' is not defined`. Also set
  `same_memchip_speed: true`, or the chiplet is clocked at 1/20 of the host clock.
- **Disk quota is the recurring hazard.** Set both trace-sink variables on any
  long run. On NFS use `truncate -s 0`, never `rm`, on a file the simulator holds
  open — `rm` silly-renames, frees nothing, and breaks the next run's cleanup.

---

## 6. What is left, in the order I would do it

1. **Re-roll the generated loops in the mini-compiler.** The DFG is statically
   unrolled, so generated host code grows linearly with task count — about 120 B
   per task. The two fixes made this summer are constant-factor wins (about 7×
   more tasks fit); re-rolling is the only change that is O(1) in task count, and
   with 128 KiB fixed for the tapeout it is what decides how large a GEMM the real
   chip can be given.
2. **Pick target configurations from real workload sizes.** Task count follows
   directly from matrix shape, so those sizes decide how far item 1 has to go.
3. **Measure the memory-chiplet bandwidth benefit** (§4). One run.
4. **Move D onto the memory chiplet.** The last on-chip term that still grows with
   matrix size is `grid_K × d_slice_size`. Moving it off-chip unlocks the next
   size tier.
5. **Revisit iDMA vs xDMA.** Everything here uses iDMA. The operand image is
   pre-arranged into K slabs by the datagen so that an inner tile is one
   contiguous run per operand, which is what makes that choice cheap. A layout
   the datagen cannot pre-arrange -- a transpose, or reading a shared full-K
   image -- would need xDMA's reshape instead.
6. **Retire `hemaia_d2d_link_initialize_4c1m()`** in favour of
   `hemaia_d2d_link_initialize()`. It is still called by every other multi_chip
   application; I left it in place with a `LEGACY, 2x2 ONLY` warning because I
   could not re-validate those applications.
