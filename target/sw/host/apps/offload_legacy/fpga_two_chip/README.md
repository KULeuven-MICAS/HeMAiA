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
before running either host.

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

Validation in this checkout compiled the device C source with the container's
RISC-V compiler and checked the packed data against the dense golden. A complete
UART image has not been linked: the existing D2D runtime references
`RX_YIELD_WINDOW`, which the hardware register description lacks; the existing
VersaCore library synchronization check also fails. The upstream checkout is
additionally missing the core-role generator required by the top-level build.
These runtime/dependency mismatches need to be resolved before the standard
build command can complete. Validation used `hemaia_tapeout_2c.hjson` directly
(the same two VersaCore clusters and 128 KiB L3 as the cached configuration),
without replacing `lru.hjson`. No FPGA execution has been performed.

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

Mailbox words have nonzero initial values in `.data`, so UART loading initializes
them and a later `_start` does not clear a peer's early notification. The host
disables its D-cache before polling because incoming D2D writes are not coherent
with cached L3 reads. Only the connected local routing port and multicast delivery
are enabled; the launcher preserves the FPGA clock/PHY setup and does not reset
the link or access the ASIC grid's memory-chip registers.

Waiting has no timeout because the second UART start can be arbitrarily late.
The protocol is one run per load: for another run, reload **both** images before
starting either host again. Reloading one image while its peer is running would
invalidate synchronization state.
