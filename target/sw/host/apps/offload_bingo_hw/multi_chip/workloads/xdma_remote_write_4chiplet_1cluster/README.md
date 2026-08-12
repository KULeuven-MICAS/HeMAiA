# xdma_remote_write_4chiplet_1cluster

The smallest workload that exercises a **cross-chiplet cluster-xDMA remote
write**: chiplet 00 cluster 0 pushes 1 KiB straight out of its own L1 into
chiplet 01 cluster 0's L1, and chiplet 01 checks what landed.

## Why a separate workload

None of the existing multi-chip tests cover this path:

| workload | what actually crosses the die boundary |
| --- | --- |
| `dma_read_from_chip00_*`, `dma_write_from_other_chip_*` | **host** xDMA / iDMA, L3↔L3 |
| `xdma_int32_add_4chiplet_1cluster` | **iDMA** for the cross-chip copy; xDMA only does the *local* add |
| **this one** | the **cluster xDMA**, L1 → *remote chiplet's* L1 |

That last path is not a plain AXI write. Before any data moves, the sending
xDMA configures the *receiving* cluster's writer through the xDMA cross-cluster
control MMIO — the top 12 KiB of the destination cluster window (cfg / grant /
finish, 4 KiB each, matching `xdma_axi_adapter`'s
`MMIO{Cfg,Grant,Finish}Offset`) — and then waits for a grant.

Those control writes are **narrow**. Crossing a chiplet boundary upsizes them to
**wide** for the D2D link, so the address map has to keep that top 12 KiB
narrow-only on the receiver. If it does not, the wide-arriving control write
lands on the cluster's *wide* slave, which has no cfg demux (that lives on the
narrow slave); it is dropped, no grant ever comes back, and the sender hangs
forever. The relevant address-map leaf is `quad_narrow_cluster_<i>_xdma_ctrl` in
[`util/occamygen/occamy.py`](../../../../../../../util/occamygen/occamy.py).

So a PASS here means the cross-chiplet xDMA cfg/grant handshake completed **and**
the payload arrived byte-exact. A hang means the handshake never closed.

## Task graph

```
Load_Payload_MemPool_to_Chip00_L1 ─┐
                                   ├─> XDMA_Remote_Write_Chip00_to_Chip01 ─> Check_Payload_Received_On_Chip01
Poison_Recv_Buffer_Chip01_L1     ──┘
```

1. Chiplet 00's DM core iDMA-loads the payload (`0xa5a50000 + i`) from the
   MemPool chip into its L1.
2. Chiplet 01's DM core iDMA-loads `0xdeadbeef` into its own L1 receive buffer,
   so a PASS cannot come from stale or coincidentally-correct L1. This edge is
   ordered *before* the write, or it would clobber the result.
3. Chiplet 00's cluster xDMA writes L1 → chiplet 01's L1. **The transfer under
   test.**
4. Chiplet 01's host core compares the receive buffer against a golden array in
   its own L3 (`xdma_rw_golden_l3`). The golden is in the ELF rather than in the
   MemPool chip so the checker itself performs no cross-chip read — otherwise a
   failure there would confound the thing being measured.

Chiplets 10 and 11 are present (the platform has four) but only run their exit
chain.

## How chiplet 00 names a buffer on chiplet 01

Bingo emits L1 allocations per chiplet in alphabetical order of handle name
(`BingoDFG._collect_memory_handles` sorts by `h.name`), so identically-named
handles preceded by the same set of names land at the same cluster-local offset.
This workload has exactly **one** L1 buffer name, `A_xfer_l1`, allocated on both
chiplets — so both sit at the same offset by construction. Chiplet 00 then takes
its *own* `ptr_A_xfer_l1`, masks off the chip tag, and re-tags it with 0x01:

```c
args_dev_chip00_2->dst_addr_lo =
    (uint32_t)(chiplet_addr_transform_full(0x01,
        ((uint64_t)(ptr_A_xfer_l1) & 0x000000ffffffffffULL)));
```

Watch out when extending this: a `BingoMemAlloc` that no node's `kernel_args`
references is never collected, so it is never allocated — adding a padding
buffer on one chiplet only, to line offsets up, silently does nothing.

## Running

```sh
cd target/sim/automation/ci/local_ci/tapeout_1c && python3 run_local_ci.py -j 8
```

It is registered in the `tapeout_1c` and `tapeout_1c_simd` suites. Expect
`payload_recv_chip01` to report a pass in the UART log. Note that a failing RTL
sim still exits 0 — read the printed verdict, not the exit code.

## Knobs

- `payload_bytes` in `params.hjson` — must be a multiple of `XDMA_WIDTH` (64 B),
  or `xdma_memcpy_1d_full_addr` refuses the transfer and `xdma_start()` hangs on
  a stale descriptor.
- `SRC_CHIPLET` / `DST_CHIPLET` in `main_bingo.py` — `0x00 -> 0x01` is a single
  D2D hop. Set `DST_CHIPLET = 0x11` for the diagonal (two-hop) case.
