# Host-Only Applications

This directory contains applications that run exclusively on the host processor.

## Concept: No Device Binaries
The "host-only" category refers to applications where `INCL_DEVICE_BINARY` is set to `0`. These programs do not contain an embedded device kernel. Instead, they interact with the hardware platform directly for system-level tasks such as:
- **System Bring-up**: Configuring clocks, resets, UART, Initializing Die-to-Die (D2D) chiplet links.
- **Communication Testing**: Verifying Host-to-Chiplet and Chiplet-to-Chiplet mailbox and synchronization primitives.
- **Data Movement**: Orchestrating system-level XDMA transfers.

## Applications

### Single-chip applications
- **hello_world**: Simple print verification of the host runtime.
- **clk_rst_configurator**: Low-level configuration of the clock and reset controller.
- **ara_\<kernel\>** (ara_add, ara_exp, ara_softmax, ...): Per-kernel cycle-count characterization for the FP32 RVV host kernels. One app per kernel, each self-contained (its own `src/main.c`), timing a single kernel and generating its test data via the shared `util/sim/ara/ara_lib.py`. Driven as a sweep by `target/sim/automation/sweep/unit_test/ara/`.
- **io_drive_strength**: Checks the configuration path of the SoC controller's `IO_DRIVE_STRENGTH` register and the `set_io_drive_strength()` helpers: every per-domain field is reachable, independently writable and reads back what was written.
- **ci_ara**: Fast CI regression checking correctness of all FP32 RVV host kernels dispatched to CVA6+Ara.
- **test_bingo_alloc**: Test suite for the `bingo_alloc` allocator.

### Multi-chip applications
- **d2d_link_configurator**: Setup and training of the die-to-die communication links.
- **d2d_simple_test**: Basic D2D verification — the host CPU issues D2D writes to the memory chip, then reads them back and checks.
- **d2d_multi_streams**: D2D multi-stream test where each chiplet in the 2x2 topology concurrently streams data.
- **d2d_broadcast_region**: Broadcasts (stores to chip `0xFF`) on a grid with memory chips between the compute chips: every compute chip must receive every broadcast and no memory chip may consume one.
- **d2d_push_pull_bw**: Pull (this chip's system iDMA reads HBM or memory-chip SRAM) against push (the memory chiplet's iDMA writes this chip's L3) over the half-duplex D2D link, with the links in SDR and in DDR; every copy is checked.
- **hbm_simple_test**: Smoke test of the memory chip's simulated HBM: image read-back, pseudo-channel stores, load latency, an iDMA pull into L3 and a memory-chip HBM-to-HBM copy, all checked.
- **memchip_multi_engine**: The memory chip's push engines, and memory chips that hold different data: every engine copies HBM to its own SRAM, then parallel and serial pushes to the compute chips are checked.
- **multichip_mailbox**: Verification of inter-chiplet communication.
- **multichip_sync**: Barrier synchronization across multiple hardware chiplets.
- **system_idma_copy**: Multi-chip system-level iDMA copy test.
- **system_xdma_copy**: Benchmarking and verifying system-level xDMA transfers.

## Build Command
```bash
# Single-chip
make single-sw HOST_APP_TYPE=host_only CHIP_TYPE=single_chip WORKLOAD=hello_world
# Multi-chip
make single-sw HOST_APP_TYPE=host_only CHIP_TYPE=multi_chip WORKLOAD=multichip_sync
```