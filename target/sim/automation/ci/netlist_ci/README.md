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

The supported profiles reuse main's categorized local-CI task lists directly:

| `--hardware` | Tapeout cfg | Categorized suite | Tasks |
| --- | --- | --- | ---: |
| `1c` | `hemaia_tapeout_1c.hjson` | `local_ci/tapeout_1c` | 36 |
| `1c_simd` | `hemaia_tapeout_1c_simd.hjson` | `local_ci/tapeout_1c_simd` | 41 |
| `2c` | `hemaia_tapeout_2c.hjson` | `local_ci/tapeout_2c` | 6 |
| `2c_simd` | `hemaia_tapeout_2c_simd.hjson` | `local_ci/tapeout_2c_simd` | 7 |

Each cfg has an explicitly named local-CI suite. Netlist CI does not maintain
separate task YAML copies, so changes to main's categorized suites automatically
reach their matching netlist profiles.

The task directories and preparation manifest share this directory as one
active handoff. Preparing any hardware profile replaces the previous one;
use the same `--hardware` value for `simulate`.

Gate-level execution defaults to one simulation process because each compiled
netlist consumes substantial memory. Use `-j` only when the backend capacity is
known.
