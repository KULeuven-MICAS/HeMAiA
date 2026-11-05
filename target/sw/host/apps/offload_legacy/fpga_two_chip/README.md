# Two-FPGA legacy GEMM

One combined host/device image runs on both boards. Each board keeps its code,
matrices, output and synchronization words in its own L3. No memory chip or
`mempool.bin` is used. The device workload is
[`gemm_fpga`](../../../../device/apps/snax/gemm_fpga).

## Wiring and IDs

| Board | Connected port | Hardware chip ID | GEMM role |
| --- | --- | --- | --- |
| Board 0 | EAST | `0x00` | First A row block, source of shared B |
| Board 1 | WEST | `0x10` | Second A row block, receives B |

These are the default VIO IDs in `hemaia_system_east` / `hemaia_system_west`.
IDs encode `(x << 4) | y`: `0x01` is a north/south neighbor and cannot replace
`0x10` for this wiring. Both FPGA bitstreams must be configured and out of reset
before running either host. Run the FPGA flow's separate D2D initialization
program first. The GEMM image inherits the configured UART, D2D links, routing,
clocks and PHY without reprogramming them.

## Build

From the repository root, in the HeMAiA toolchain environment:

```sh
make single-sw HOST_APP_TYPE=offload_legacy CHIP_TYPE=fpga_two_chip DEV_APP=gemm_fpga
```

This uses the current `target/rtl/cfg/lru.hjson`. Set
`CFG_OVERRIDE=/path/to/the/bitstream/config.hjson` when selecting a different
hardware configuration. The configuration must match the FPGA's memory map,
cluster count and VersaCore/streamer hardware. The checked-in
`hemaia_vpk180.hjson` describes an ALU cluster, so it is not a VersaCore GEMM
configuration. This app uses cluster 0 for the workload on each board.

After a successful build, the UART image is:

```text
target/sw/host/apps/offload_legacy/fpga_two_chip/build/offload_legacy_fpga_two_chip_gemm_fpga.bin
```

The corresponding `.elf` contains the host symbols. The device symbols are in
`target/sw/device/apps/snax/gemm_fpga/build/offload_legacy_fpga_two_chip_gemm_fpga.elf`.
The final host link checks that the embedded image fits the configured L3.
The FPGA app is opt-in and is not added to the generic software fleet build.

The hardware configuration may describe a larger ASIC grid: for two connected
FPGAs at `00` and `10`, select `CHIP_TYPE=fpga_two_chip` for the two-board host
handshake. Keep the hardware configuration matching the
bitstreams, including the cluster count and streamer layout.

Validation in the HeMAiA container completed `make apps` for both
`CHIP_TYPE=multi_chip` and `CHIP_TYPE=fpga_two_chip`, with
`HOST_APP_TYPE=offload_legacy DEV_APP=gemm_fpga` and the current
`hemaia_16c4m_8MBL3_2cluster` configuration (`snax_split_cluster`). Both complete
host/device images linked successfully. The packed-data checks also passed
against both `snax_split_cluster` and `snax_versacore_to_cluster`, including
three-loop C/D layouts. No FPGA execution has been performed.

## Load, then run

1. On each board, use boot-ROM option **3** to load the same combined `.bin`
   into that board's L3. Keep the boot-ROM target chip ID equal to its local ID.
2. After **both** loads finish, select option **7** on either board. Its host
   waits for the other board.
3. Select option **7** on the second board. The hosts exchange READY, GO and ACK,
   then each host wakes its own Snitch clusters.
4. Check the GEMM result and host exit code on both UARTs. A zero exit code
   indicates the result checks passed on both boards.

The host handshake is explicit in
[`src/offload_legacy_fpga_two_chip.c`](src/offload_legacy_fpga_two_chip.c):

```text
0x10: finish local setup -> write READY to 0x00 -> wait for GO -> write ACK -> run
0x00: finish local setup -> wait for READY -> write GO to 0x10 -> wait for ACK -> run
```

This initial rendezvous precedes device execution. The shared `chip_barrier`
uses checkpoints in `comm_buffer` and broadcasts them; a late-starting host
clears that buffer during initialization and can erase an earlier arrival.
The dedicated host handshake instead uses flags initialized by loading the
image, which neither host clears on entry. Runtime chip barriers can still
be used after both hosts have finished initialization.

Mailbox words have nonzero initial values in `.data`, so UART loading initializes
them and a later `_start` does not clear a peer's early notification. The host
temporarily disables its D-cache for the handshake, then restores its previous
state before waking Snitch. Both narrow SPM (`0x7000...`) and wide SPM
(`0x8000...`) are cached regions in the current CVA6 wrapper. The cache-control
CSR is `0x7c1`; writing the obsolete `0x701` traps on this RTL.

Waiting has no timeout because the second UART start can be arbitrarily late.
The protocol is one run per load: for another run, reload **both** images before
starting either host again. Reloading one image while its peer is running would
invalidate synchronization state.

The first application message is `[HeMAiA] Two-FPGA legacy offload, chip ...`.
Further messages identify local Snitch preparation, entry into the host
handshake, and completion with the original D-cache state restored.
