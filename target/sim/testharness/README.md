# HeMAiA Simulation Testharness

This directory contains the simulation testharness for the HeMAiA chip.
All SystemVerilog sources under `testharness/` and `testharness/util/` are
**generated** from Mako templates in `template/` — do not edit them by hand.

## Architecture Overview

The testharness mirrors a real testing setup with three layers:

```
testharness  (top-level, drives CLK/RST/PLL/peripherals, loads binary, monitors finish)
  │
  ├── dut  (Design Under Test: compute chiplets + io_wrapper)
  │     │
  │     ├── hemaia instances  (one per compute chiplet position)
  │     │
  │     └── io_wrapper  (D2D interconnect routing between chiplets and off-chip)
  │           ├── gen_direct       (SIM_WITH_INTERPOSER=0, ideal wire model)
  │           └── gen_interposer   (SIM_WITH_INTERPOSER=1, IO-pad + interposer model, 2x2 only)
  │
  └── mem_chip instances  (external memory pool, FPGA in real setup, simulated here)
```

### testharness (top level)

Mimics the real board-level test setup. Responsibilities:

1. **Clock generation** — drives `mst_clk_i` and `periph_clk_i`
2. **Reset sequencing** — assert then release reset
3. **PLL control** — if vendor PLL is present, enable and wait for lock
4. **Binary loading** — calls `load_binary()` to preload SRAM banks via backdoor
5. **Finish monitoring** — calls `check_finish()`, which watches a sentinel memory
   location in each chiplet's SRAM; software writes `1` for pass, anything else for fail

Each compute chiplet gets its own UART (via `uartdpi`), JTAG, I2C, SPI, and GPIO
peripherals wired at this level.

### dut (Design Under Test)

A structural module (no behavioral code) that instantiates:

- **Compute chiplets** (`hemaia` instances) — one per grid position `(x, y)`.
  Each chiplet exposes 4 D2D interfaces (north/south/east/west) plus peripheral pins.
- **io_wrapper** — receives all chiplet D2D ports and decides how to route them.

The `dut` cleanly separates chip RTL from the interconnect fabric so that different
physical routing models can be selected without changing chiplet instantiation.

### io_wrapper (interconnect routing)

Handles all die-to-die (D2D) signal routing. Two modes selected by `SIM_WITH_INTERPOSER`:

| Mode | `SIM_WITH_INTERPOSER` | Description |
|------|-----------------------|-------------|
| **Direct** | `0` | Ideal wire connections. Adjacent chiplets' D2D buses are shorted via `tran` gates; flow-control signals use `assign`. No physical effects. |
| **Interposer** | `1` | Each chiplet is wrapped in `hemaia_io_pad` (IO pad model), then routed through `hemaia_chiplet_interconnect` instances modeling the silicon interposer. **Currently supports 2x2 arrays only.** |

Internally, the io_wrapper connects:
- **Internal mesh** — adjacent chiplets' facing D2D ports (e.g., chip `(0,0)` east ↔ chip `(1,0)` west)
- **Boundary** — boundary chiplets' outward D2D ports to off-chip module ports (connected to memory chips at testharness level)

### Memory chips

Memory chiplets (`hemaia_mem_chip`) sit around the compute chiplet array and
communicate via the off-chip D2D links. In the real setup, these will be
implemented on an FPGA; in simulation, they are behavioral models.

```
              north boundary (off-chip mem slots)
          +--------+--------+--------+
  west    | (0,0)  | (1,0)  | (2,0)  |   east
  mem     +--------+--------+--------+   mem
  slots   | (0,1)  | (1,1)  | (2,1)  |   slots
          +--------+--------+--------+
              south boundary (off-chip mem slots)
```

## Generated Files

| Output | Template | Description |
|--------|----------|-------------|
| `testharness.sv` | `template/testharness.sv.tpl` | Top-level testharness |
| `dut.sv` | `template/dut.sv.tpl` | DUT: chiplet instances + io_wrapper |
| `io_wrapper.sv` | `template/io_wrapper.sv.tpl` | D2D routing (direct / interposer) |
| `util/load_binary.sv` | `template/load_binary.sv.tpl` | Backdoor binary loading tasks |
| `util/check_finish.sv` | `template/check_finish.sv.tpl` | Simulation finish monitor |

## Regenerating

From `target/sim/`:

```bash
# Normal (silent) generation:
make testharness/testharness.sv

# Debug (verbose) generation:
make debug-testharness-gen
```

Key Make variables that control generation:

| Variable | Default | Effect |
|----------|---------|--------|
| `SIM_WITH_INTERPOSER` | `0` | Use interposer routing model |
| `SIM_WITH_PLL` | auto from `CFG` | Include vendor PLL |
| `SIM_WITH_MACRO` | `0` | Use vendor SRAM macros |
| `SIM_WITH_NETLIST` | `0` | Use post-synthesis netlist |
