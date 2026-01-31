# Host-Only Applications

This directory contains applications that run exclusively on the host processor.

## Concept: No Device Binaries
The "host-only" category refers to applications where `INCL_DEVICE_BINARY` is set to `0`. These programs do not contain an embedded device kernel. Instead, they interact with the hardware platform directly for system-level tasks such as:
- **System Bring-up**: Configuring clocks, resets, UART, Initializing Die-to-Die (D2D) chiplet links.
- **Communication Testing**: Verifying Host-to-Chiplet and Chiplet-to-Chiplet mailbox and synchronization primitives.
- **Data Movement**: Orchestrating system-level XDMA transfers.

## Applications

- **hello_world**: Simple print verification of the host runtime.
- **clk_rst_configurator**: Low-level configuration of the clock and reset controller.
- **d2d_link_configurator**: Setup and training of the die-to-die communication links.
- **multichip_mailbox**: Verification of inter-chiplet communication.
- **multichip_sync**: Barrier synchronization across multiple hardware chiplets.
- **system_xdma_copy**: Benchmarking and verifying system-level XDMA transfers.

## Build Command
```bash
make single-sw HOST_APP_TYPE=host_only CHIP_TYPE=single_chip WORKLOAD=hello_world
```