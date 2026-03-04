# Legacy Offload Applications

This directory contains "legacy" offload applications that follow a traditional host-device interaction model.

## Concept: Explicit SOC Control
In the legacy model, the host performs the following steps:
1.  **Binary Loading**: The device binary is embedded within the host binary and loaded into the target memory (e.g., L3) during system startup.
2.  **Metadata Setup**: The host manually writes the device's entry point to the `soc_ctrl` scratch registers and the device will then jmp to the entry point.
3.  **Wakeup**: The host releases the device clusters from reset or wakes them up via SOC control signals.
4.  **Synchronization**: The host polls for a completion flag or waits for a specific interrupt to know when the device has finished its task.

## Supported Device kernels (DEV_APP)
The legacy host application (`offload_legacy_single_chip.c`) can be linked with various device apps:
- **snax-test-integration**: Basic verification of the SNAX accelerator interface.
- **snax-xdma-copy**: Verification of XDMA engines within the snitch clusters.
- **snax-xdma-hazard**: Stress test for memory hazards and consistency.
- **snax-versacore-matmul-profile**: Matrix multiplication implementation running on the Versacore functional unit.

## Build Command
```bash
make single-sw HOST_APP_TYPE=offload_legacy CHIP_TYPE=single_chip DEV_APP=snax-test-integration
```
