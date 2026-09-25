# Netlist CI

This entrypoint uses only fabricated hardware profiles. It always selects
`sim_netlist.hjson`, mapped SRAM macros, D2D, and the vendor PLL. The main memory
is fixed at 128 KiB: 16 banks of 1024 64-bit words. There are intentionally no
configuration or memory-size override options.

```sh
# Start from a full clean performed in the HeMAiA build image, then initialise
# the private modules on the host. The preparation runner also executes every
# software, boot-ROM, and RTL build command in this image.
podman run --rm -v "$PWD:$PWD" -w "$PWD" \
  ghcr.io/kuleuven-micas/hemaia:main make clean
target/tapeout/1_git_pull_private_modules.sh --macro=1 --d2d=1 --pll=1
target/tapeout/HeMAiAv2_tapeout/helper_shell_script/6.1_prepare_netlist_ci.sh \
  --hardware 1c

# On the EDA backend, combine that handoff with the raw mapped netlist and its
# step-5 profile. The helper validates their hardware/source identity, comments
# the duplicate embedded boot ROM, compiles, and runs one simulator.
target/tapeout/HeMAiAv2_tapeout/helper_shell_script/6.2_compile_run_netlist_ci.sh \
  --hardware 1c --engine vcs
```

By default, simulation replaces the synthesized boot ROM with the generated
RTL boot ROM. To use the original boot ROM embedded in the mapped netlist, add
`--use-original-bootrom` to both preparation and backend simulation:

```sh
python3 -u target/sim/automation/ci/netlist_ci/run_netlist_ci.py \
  --hardware 2c --phase prepare --waveform 0 --use-original-bootrom

target/tapeout/HeMAiAv2_tapeout/helper_shell_script/6.2_compile_run_netlist_ci.sh \
  --hardware 2c --engine vcs --use-original-bootrom \
  --netlist /path/to/hemaia_mapped.v
```

The `6.1_prepare_netlist_ci.sh` helper also accepts this flag. Preparation omits
the RTL boot ROM from both simulator file lists; step 6.2 copies the raw netlist
unchanged to `outputs/hemaia_mapped_original_bootrom.v`. This uses the ROM
contents fixed at synthesis. The preparation manifest records the selection
and rejects a simulation request with a different selection. Existing handoffs
without a boot ROM setting retain the default RTL replacement behavior.
The original ROM also retains its synthesized boot behavior: the checked-in
`bootrom_chip` firmware waits for a UART menu selection (`7` to continue booting).

To launch just one workload from an existing full-suite preparation, add
`--task NAME` only to step 6.2; no new preparation is needed. For example,
`io_drive_strength` is a small host-only register test that can
provide the application image while you interact with the original boot ROM:

```sh
python3 -u target/sim/automation/ci/netlist_ci/run_netlist_ci.py \
  --hardware 2c --phase prepare --waveform 0 \
  --use-original-bootrom

target/tapeout/HeMAiAv2_tapeout/helper_shell_script/6.2_compile_run_netlist_ci.sh \
  --hardware 2c --engine vcs --netlist /path/to/hemaia_mapped.v \
  --use-original-bootrom --task io_drive_strength --max-sim-jobs 1
```

To build only one workload in a new preparation, pass the same `--task NAME`
to preparation and simulation. The `6.1_prepare_netlist_ci.sh` helper also
accepts `--task NAME`. Names can be
the full CI task name, or a workload/device application name that matches
exactly one task in the selected hardware profile. List the full names without
building or launching anything:

```sh
python3 target/sim/automation/ci/netlist_ci/run_netlist_ci.py \
  --hardware 2c --list-tasks
```

Omitting `--task` retains the full suite. `--max-sim-jobs 1` only sets concurrency;
it does not limit the total number of tasks. Simulation can select any task
already present in the preparation manifest, and keeps that task's existing
directory and index. For `2c`, `io_drive_strength` uses
`task_3_host_only_single_chip_io_drive_strength` after full-suite preparation,
or `task_0_host_only_single_chip_io_drive_strength` after preparation with that
task alone. To run tasks that were not prepared, re-run preparation with the
desired selection (omit `--task` for the full suite). Preparation replaces the
active handoff and its task directories. Keep the same hardware and boot ROM
settings between preparation and simulation.

One task launches one simulator containing all compute chips in the hardware
profile. Each chip's original ROM still waits for UART input; task selection
does not automatically send `3` or `7`. If you send `7`, the preloaded workload
starts. For VCS, each task's `bin/transcript` is the live simulator log, and
`bin/uart_chip_X_Y.log` records the corresponding chip's UART output.

The supported profiles reuse main's categorized local-CI task lists directly:

| `--hardware` | Tapeout cfg | Categorized suite | Tasks |
| --- | --- | --- | ---: |
| `1c` | `hemaia_tapeout_1c.hjson` | `local_ci/tapeout_1c` | 36 |
| `1c_simd` | `hemaia_tapeout_1c_simd.hjson` | `local_ci/tapeout_1c_simd` | 41 |
| `2c` | `hemaia_tapeout_2c.hjson` | `local_ci/tapeout_2c` | 14 |
| `2c_simd` | `hemaia_tapeout_2c_simd.hjson` | `local_ci/tapeout_2c_simd` | 15 |

Each cfg has an explicitly named local-CI suite. Netlist CI does not maintain
separate task YAML copies, so changes to main's categorized suites automatically
reach their matching netlist profiles.

The task directories and preparation manifest share this directory as one
active handoff. Preparing any hardware profile replaces the previous one;
use the same `--hardware` value for `simulate`.

Gate-level execution defaults to one simulation process because each compiled
netlist consumes substantial memory. Use `-j` only when the backend capacity is
known.
