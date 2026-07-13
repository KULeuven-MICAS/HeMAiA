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

The supported profiles reuse main's categorized local-CI task lists directly:

| `--hardware` | Tapeout cfg | Categorized suite | Tasks |
| --- | --- | --- | ---: |
| `1c` | `hemaia_tapeout.hjson` | `local_ci/tapeout` | 36 |
| `1c_simd` | `hemaia_tapeout_1c_simd.hjson` | `local_ci/tapeout_1c_simd` | 5 |
| `2c` | `hemaia_tapeout_2c.hjson` | `local_ci/tapeout_2c` | 6 |
| `2c_simd` | `hemaia_tapeout_2c_simd.hjson` | `local_ci/tapeout_2c_simd` | 1 |

The historical plain `hemaia_tapeout.hjson` is the `1c` profile. Netlist CI
does not maintain separate task YAML copies, so changes to main's categorized
suites automatically reach their matching netlist profiles.

The task directories and preparation manifest share this directory as one
active handoff. Preparing any hardware profile replaces the previous one;
use the same `--hardware` value for `simulate`.

Gate-level execution defaults to one simulation process because each compiled
netlist consumes substantial memory. Use `-j` only when the backend capacity is
known.
