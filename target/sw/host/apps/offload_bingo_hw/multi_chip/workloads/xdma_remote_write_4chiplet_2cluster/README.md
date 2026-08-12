# xdma_remote_write_4chiplet_2cluster

The extended form of
[`xdma_remote_write_4chiplet_1cluster`](../xdma_remote_write_4chiplet_1cluster/README.md):
chiplet 00 cluster 0 pushes the same 1 KiB payload out of its own L1 into **both**
clusters of chiplet 10, and chiplet 10 checks both.

## What this adds over the 1-cluster workload

The 1-cluster workload proves the cross-chiplet xDMA cfg/grant handshake closes
for *one* destination cluster. It cannot say anything about *which* cluster on
the far side was addressed, because there is only one.

The destination cluster is not a field in the transfer — it is implied by the
destination **address**. The sending xDMA derives the receiving cluster's control
MMIO (cfg / grant / finish, the top 12 KiB of that cluster's window) from the
address it was handed, so a cluster-1 destination exercises a *different* control
window than a cluster-0 one. Everything downstream of that derivation — the
narrow-only routing of the control region, the grant coming back from the right
cluster, the finish matching the right task — is re-run against a second window.

A collision is caught by construction: both destinations are poisoned first and
both are checked, so if the cluster-1 address resolved back onto cluster 0 (or
vice versa), the neglected cluster stays poisoned and its check fails.

## Task graph

```
Load_Payload_MemPool_to_Chip00_C0_L1 ─┐
                                      ├─> XDMA_Remote_Write_Chip00_C0_to_Chip10_C0 ─> Check_..._Chip10_C0
Poison_Recv_Buffer_Chip10_C0_L1     ──┘                 │
                                                        ├─> XDMA_Remote_Write_Chip00_C0_to_Chip10_C1 ─> Check_..._Chip10_C1
Poison_Recv_Buffer_Chip10_C1_L1     ──────────────────┘
```

The two writes are serialized so the sender issues them in a known order; the
payload dependency reaches the second write transitively. Chiplets 01 and 11 are
present (the platform has four) but only run their exit chain.

## Addressing the second cluster

Chiplet 00 must name a buffer that lives on chiplet 10, in a cluster it is not
running on. Two facts make that constructible:

- Bingo emits every **named** L1 handle before any scratchpad or kernel-arg
  allocation, and each `(chip, cluster)` has its own allocator starting at the
  same cluster-local base. This workload puts exactly **one** named L1 handle, of
  the same size, in each of the three `(chip, cluster)` pairs it uses — so all
  three are that allocator's first allocation and land at the same cluster-local
  offset.
- Clusters are a replicated address space: cluster *N*'s window is cluster 0's
  plus `N * cluster_offset`, and `occamygen` asserts that offset is common to all
  clusters.

So the sender takes its own `ptr_A_xfer_l1_c0`, masks off the chip tag, adds
`cluster_id * cluster_offset`, and re-tags with chiplet 10's ID:

```c
args_dev_chip00_5->dst_addr_lo = (uint32_t)(chiplet_addr_transform_full(0x10,
    (((uint64_t)(ptr_A_xfer_l1_c0) & 0x000000ffffffffffULL)
     + (uint64_t)cluster_offset * 1)));
```

### Why the handles are *not* all called the same thing

The 1-cluster workload gives both its buffers one shared name, `A_xfer_l1`. Do
not copy that here. `_collect_memory_handles` emits one C variable `ptr_<name>`
per handle into a single **per-chip** scope, so two same-named handles on the
*same* chip — exactly what one shared name for both of chiplet 10's clusters
would produce — emit a duplicate definition and the app does not compile:

```c
uint64_t ptr_A_xfer_l1 = bingo_l1_alloc(0x10, 1, 1024);
uint64_t ptr_A_xfer_l1 = bingo_l1_alloc(0x10, 0, 1024);   // redefinition
```

The 1-cluster workload gets away with it only because its two handles sit on
different chips, hence in different scopes. Sharing a name is also not what makes
the offsets line up — being each cluster's sole named allocation is.

Watch out when extending this: a `BingoMemAlloc` that no node's `kernel_args`
references is never collected, so it is never allocated — adding a padding buffer
to line offsets up silently does nothing.

## Running

```sh
cd target/sim/automation/ci/local_ci/tapeout_2c_simd && python3 run_local_ci.py -j 8
```

Registered in the `tapeout_2c_simd` suite. Expect `payload_recv_chip10_c0` and
`payload_recv_chip10_c1` to both report a pass in chiplet 10's UART log. Note
that a failing RTL sim still exits 0 — read the printed verdict, not the exit
code.

## Knobs

- `payload_bytes` in `params.hjson` — must be a multiple of `XDMA_WIDTH` (64 B),
  or `xdma_memcpy_1d_full_addr` refuses the transfer and `xdma_start()` hangs on
  a stale descriptor.
- `SRC_CHIPLET` / `DST_CHIPLET` / `DST_CLUSTERS` in `main_bingo.py`. `0x00 ->
  0x10` is a single D2D hop; `0x11` would be the diagonal, two-hop case.
- The payload, poison and golden come from the 1-cluster workload's
  `xdma_remote_write_datagen.py`, imported rather than copied so the two cannot
  drift. The Makefile depends on that file.
