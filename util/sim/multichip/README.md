# util/sim/multichip: what every multi-chiplet workload generator shares

The generators of the blocks (`target/sw/host/apps/offload_bingo_hw/multi_chip/workloads/{ffn,fa,moe}`) and of the
layers built from them (`workloads/{llama3,dsv2}` there) import these modules by name. A block's driver puts this
directory and `../common` (its `driver.py`) at the front of `sys.path`, then `mini_compiler`:

    sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "multichip"))
    sys.path.insert(0, os.path.join(os.path.abspath(__file__).split("/target/sw/host/")[0], "util", "sim", "common"))
    from driver import repo_root, run_driver

| module | what it holds |
|---|---|
| `fabric.py` | `FabricBase`: the mesh, a local buffer's address on another chip, the fold order over the compute chips |
| `chiplet_mc.py` | `ChipletBuild(FabricBase)`: the platform and its params, the weight rings, the stamps, the barriers, the broadcast of a token, the chips' fold and the reduce, the checks and `finish()` (emission). A block's build subclasses it: `ffn_prefill_mc.PBuild`, the FA, MoE and layer builds |

What is not multi-chiplet code lives with the rest of `util/sim`. A block's data module puts `../llm` (and `../gemm`
where it needs `gemv_utils`) on `sys.path`; `model_data` puts `../common` and `../simd` there:

| where | what |
|---|---|
| mini_compiler's `passes/bingo_args_phases.py` | `split_args_phases`: each cluster's task args in phases through two alternating L1 regions, for blocks whose args outgrow L1 |
| `../common/driver.py` | the checkout's root, the command line every `main_bingo.py` takes, and `run_driver`, the body of a driver's `run()` |
| `../llm/` | `model_data.py`: `import_golden` (the golden's modules), `read_bf16` (a BF16 tensor of the weight caches), `RMS_EPS`; DeepSeek-V2-Lite's constants (`dsv2_model.py` + `config.json`) and float reference (`reference.py`) |
| `../common/` | the device's arithmetic bit for bit: `fp.py`, `hwmodel.py`, `layout.py`, `pack.py` |
| `../simd/` | `simd.py`: the SIMD block's FP32 arithmetic; `run_driver` points it at the cluster cfg it builds for |
| `../gemm/gemv_utils.py` | `exact_gemv` (`hwmodel.gemv` bit for bit on float64 BLAS) and `a_row_bytes` (a GEMV's x layout) |

The golden's modules are snax_cluster's `target/snitch_cluster/sw/apps/dsv2/util` (`../llm/model_data.py` says what
differs).

Every block's build (`ffn.mk`, `fa.mk`, `moe.mk`, `llama3.mk`, `dsv2_prefill.mk`) lists these files, `driver.py` and the
golden's among the generator's sources, so an edit to any of them makes every multi-chip workload regenerate at its
next build.
