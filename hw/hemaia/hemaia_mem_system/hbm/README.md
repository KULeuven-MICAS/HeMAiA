# Simulated HBM on the memory chip

The memory chip (`hemaia_mem_chip`) stands in for the FPGA next to the HeMAiA
chiplets, and that FPGA has HBM. Its 64 MiB SRAM is too small for LLM weights, so
the memchip's DRAM port (wide xbar port 3, which was tied off) now leads to
`hemaia_hbm_model`: an AXI4 slave holding GiBs, with the latency and channel
parallelism of an FPGA HBM controller. Simulation only, like the memchip itself.

| File | What |
|---|---|
| `hemaia_hbm_pkg.sv` | `hbm_cfg_t` and the default configuration |
| `hemaia_hbm_model.sv` | the AXI slave and the timing model |
| `hemaia_hbm_dpi.cc` | the storage (DPI-C) |
| `test/run.sh` | unit test of the model on its own (VCS, ~30 s) |

## Address map

The HBM is chip-local `0x1_0000_0000` up to `base + size` on each memory chip
(16 GiB by default: `0x1_0000_0000`..`0x5_0000_0000`). From a compute chip:

```c
uint64_t hbm = bingo_get_hbm_base(memchip_x, memchip_y);   // libbingo
// same as chiplet_addr_transform_loc(memchip_x, memchip_y, HBM_BASE_ADDR)
volatile uint64_t *w = (volatile uint64_t *)(hbm + offset);
```

`HBM_BASE_ADDR` and `HBM_SIZE` come from the cfg through `occamy_memory_map.h`
(`HBM_SIZE` is 0 when the memchip has no HBM). Anything that issues plain AXI to
the address reaches it: CPU loads/stores, the system iDMA of a compute chip or of
the memchip, a cluster's DMA. The HBM is not heap-managed: a workload places its
data at fixed offsets, the same offsets its manifest loads.

An access outside `[base, base + size)` is answered with SLVERR (a CVA6 load/store
there traps) and touches nothing.

## Loading data

A workload that wants data in the HBM writes, into its build directory,

```
build/hbm/manifest.txt
build/hbm/<files the manifest lists>
```

The manifest has one entry per line; `#` starts a comment:

```
# offset above the HBM base   file                 [chip=<id>]
0x0                           layer0_weights.bin
0x4000_0000                   layer1_weights.bin
0x2_0000_0000                 kv_cache_init.bin    chip=0x20
```

* `offset` is relative to the HBM base (C syntax, `_` separators allowed).
* `file` is raw bytes, relative to the manifest's directory unless absolute.
* An entry applies to every memory chip, unless `chip=<id>` names one (chip id =
  `(x << 4) | y`).
* Files must not overlap. Bytes nothing loaded read as zero.

`make apps` (`target/sim/apps/Makefile`) links `build/hbm/` into
`target/sim/bin/hbm/`, and the sim runner carries it into each task's `bin/`.
Both keep symlinks: an image is never copied. At time zero the testharness
(`load_hbm` in `target/sim/testharness/template/load_binary.sv.tpl`) clears every HBM and loads
`hbm/manifest.txt`; `+hbm_manifest=<path>` loads another one. No manifest, no
error: the HBM starts zeroed.

Files are `mmap`ed copy-on-write, not read: a GiB image loads instantly, host
memory is spent only on the pages the simulation touches, and writes from the RTL
never reach the file on disk.

`hbm_simple_test` (`target/sw/host/apps/host_only/multi_chip/`) is a worked
example: its `data/datagen.py` writes the manifest and the header from one list.

## Configuration

A memory chip has an HBM when its entry in the platform cfg
(`target/rtl/cfg/*.hjson`, `hemaia_multichip.testbench_cfg.hemaia_mem_chip`) has an
`hbm` object; without one, the DRAM port stays tied off. Today only
`hemaia_twochiplet_16MBL3_4cluster.hjson` (one compute chip, the memory chip east of it
at [1,0]) has one, spelling the full default configuration out; change a field there:

```hjson
hemaia_mem_chip: [
  {
    coordinate: [1,0],
    mem_size: 67108864,
    hbm: {
      base: 4294967296,     // 0x1_0000_0000
      size: 34359738368,    // 32 GiB instead of 16
      num_channels: 32,
      ...
    }
  },
]
```

`docs/schema/hemaia_hbm.schema.json` (referenced from `occamy.schema.json`) documents
every field and rejects unknown keys and out-of-range values. A field left out of the
object takes the schema's default below.

| Key | Default | Meaning |
|---|---|---|
| `base` | `0x1_0000_0000` | chip-local base; the DRAM port starts here, so not lower |
| `size` | `0x4_0000_0000` (16 GiB) | capacity |
| `num_channels` | 32 | pseudo-channels: independent command + data buses |
| `num_banks` | 16 | banks per pseudo-channel |
| `row_bytes` | 1024 | row (page) size per pseudo-channel |
| `interleave_bytes` | 4096 | bytes per channel before the next; 0 = one contiguous range per channel |
| `channel_mbps` | 25600 | peak MB/s of one pseudo-channel |
| `t_ctrl_ps` | 90000 | controller + PHY + fabric, request and response together |
| `t_cl_ps` / `t_cwl_ps` | 14000 / 7000 | column read / write to data |
| `t_rcd_ps` / `t_rp_ps` | 14000 / 14000 | activate / precharge |
| `t_refi_ps` / `t_rfc_ps` | 3900000 / 350000 | refresh interval / duration; `t_refi_ps: 0` disables |
| `max_reads` / `max_writes` | 64 / 64 | bursts in flight before back-pressure |

The defaults are HBM2e as an AMD FPGA exposes it: two stacks of 16 pseudo-channels
at 25.6 GB/s each (819 GB/s). The latency is calibrated to a measured FPGA HBM
(Shuhai, FCCM 2020: 106.7 ns row hit, 122.2 ns closed row, 137.8 ns conflict); the
model gives 106.5 / 120.5 / 134.5 ns. occamygen (`get_mem_chip_hbm_cfg` in
`util/occamygen/occamy.py`) renders the object into the testharness as a
`hemaia_hbm_pkg::hbm_cfg_t`; `DefaultHbmCfg` in that package repeats the schema
defaults only for an instance given no cfg.

## Timing model

All times are in picoseconds of simulated time, so latency does not change when
the memchip clock does.

* An address picks its pseudo-channel by `interleave_bytes` interleaving, then a
  bank and a row within it (row-bank-column).
* Each beat of a burst is scheduled on its channel when the burst is accepted
  (reads) or when its data arrives (writes), first come first served:
  command at `max(arrival + t_ctrl/2, bank free)`, pushed past a refresh;
  `+ t_rp + t_rcd` for a row conflict, `+ t_rcd` for a closed row; data at
  `max(command + t_cl, channel bus free)` for one beat time; done
  `t_ctrl/2` later.
* Different channels, and different banks of one channel, overlap; beats on one
  channel share its data bus.
* Refresh blocks a channel for `t_rfc` every `t_refi` (staggered across channels)
  and closes its rows.
* AXI: responses leave in order of completion across IDs and in request order
  within an ID. A read burst, once started, is not interleaved with another.

With the testbench clocks the memchip bus (512 bit at the memchip's `clk_host`)
and the D2D link, not the HBM, bound bandwidth: the HBM adds latency (~110 ns per
access) and makes many bursts in flight overlap. The channel model starts to
matter once the memchip side is made faster.

## Observability

* `+hbm_trace` prints every burst: accepted, and completed with its latency.
* At the end of a simulation each HBM that saw traffic prints bursts, beats,
  average/maximum latency, row hit/closed/conflict counts, per-channel balance,
  refresh stalls, achieved bandwidth and SLVERRs.
* `dump(path, offset, len)` on the model instance
  (`i_hemaia_mem_chip_<x>_<y>.gen_hbm.i_hbm`) writes HBM contents to a file.
