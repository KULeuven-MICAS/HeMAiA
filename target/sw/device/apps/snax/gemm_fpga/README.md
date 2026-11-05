# Two-FPGA M-split GEMM

Pair this device app with `offload_legacy/fpga_two_chip`. Load the same complete
host/device image into **both** FPGAs before starting either one. Hardware chip
IDs are `0x00` (east port) and `0x10` (west port), because IDs encode `(x << 4) | y`.
Neither this workload nor its data generator accesses a memory chip.

`D = A x B` is partitioned along M: chip 00 computes the first half of A's rows
and chip 10 the second half. Shared B is broadcast from chip 00's L3 to chip
10's L3 using a real D2D DMA transfer. With two FPGAs, this is one unicast copy
to the sole receiver. Each cluster 0 then loads its A block and B from its own
L3, runs VersaCore, and stores its result back to its own L3. Other clusters
wait in the runtime's exit reduction.

All inputs, the independent golden result, receive buffers and result buffers
are embedded in the image. The data is in the accelerator's blocked format:
`A[m,k,row,inner]`, `B[n,k,col,inner]`, `D[m,n,row,col]`. `data/params.hjson`
contains **tile counts per FPGA**. The default `array_shape=2` is bank-aligned
on both split and tapeout VersaCore clusters. On `snax_split_cluster`, its shape
is `(4,4,32)`; with `M=N=K=2`, the complete GEMM is `(16 x 8) * (8 x 64)` and
each FPGA produces an `8 x 64` INT32 result from INT8 inputs. Matrix dimensions
follow the selected cluster's shape table. C is zero and its streamer is disabled.
The UART log prints the local L3 output address, comparison result and cycles.

The device handshake exchanges READY, B_SENT/B_ACK, then one-word final status.
The DMA core publishes flags from a dedicated L1 word using full-width DMA
addresses; all polling reads local L3. The explicit nonzero mailbox and buffer
initializers belong to `.data`, so a peer's early write survives delayed device
startup. Main never clears the mailbox. **Reload both images before every run**;
restarting an already completed image without reloading is not supported.

The generator reads the platform's first cluster configuration, including its
spatial stride nests and channel widths. `FPGA_HWCFG=/path/to/cluster.hjson`
overrides this selection for standalone data generation. Run the generator
checks with:

```sh
python3 data/check_datagen.py --hwcfg /path/to/snax_versacore_to_cluster.hjson
```

The data occupies local L3; only the working operands/results and notification
word use L1. This intentionally small bring-up workload does not overlap DMA
with computation or attempt to use every cluster.
