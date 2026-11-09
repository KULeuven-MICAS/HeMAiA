# E3 synchronisation: the cost of one cross-chip synchronisation, four mechanisms

Measured on `hemaia_16c4m_8MBL3_2cluster` (the paper platform: 5 x 4 package, 16 compute chiplets in
columns x = 0, 1, 3, 4, four memory chiplets in column x = 2; clusters at 500 MHz, hosts at 1 GHz),
on the standing simv (VCS, single core).

| command | does |
|---|---|
| `PARALLEL=1 ./run.sh launch [ab] [c] [d]` | one run per arm (default all three), each with its watchdog -> `runs/stg_sync_<arm>[_<n>]` |
| `./run.sh parse runs/<ab> runs/<c> runs/<d>` | the UARTs -> `results/four_arm_sync_latency.csv` |
| `./run.sh` | re-plots `figs/fig_e3_four_arm.{pdf,png}` from the CSV |

## The four mechanisms

| arm | mechanism | how a dependency is enforced | run |
|---|---|---|---|
| A | software barrier | point-to-point stores: the master collects arrivals and pushes the release | `ab`: `offload_legacy`, `snax-test-chip-barrier` |
| B | 0xFF broadcast barrier | one store, multicast by the D2D router | `ab` (same binary, second phase) |
| C | BINGO software manager | host-mediated: mailbox, then manager, then dispatch, per edge | `c`: `offload_bingo_sw`, `sync_latency_sweep_1cluster` |
| D | BINGO hardware manager | the same graph, resolved in RTL | `d`: `offload_bingo_hw`, `sync_latency_sweep_1cluster` |

A and B are barriers: everyone waits for everyone. C and D are per edge: a task waits only for the
producers it declares. A and B are measured as one barrier (the average of 32 back to back, averaged
over the participants); C and D as a round trip from chip 0x00 (two edges), for a 1-to-1 and a 1-to-all
edge. `mcycle` is not synchronised across chiplets, so every C and D stamp is taken on chip 0x00.

## The participants

Each P is a rectangle anchored at chip 0x00, and P counts its **compute** chips; a memory chiplet inside a
rectangle runs no code and gets no task (`HEMAIA_IS_COMPUTE_CHIP`).

| P | rectangle | chips | furthest chip, hops |
|---:|---|---|---|
| 2 | 0x00-0x01 | (0,0) (0,1) | 0x01, 1 |
| 4 | 0x00-0x11 | x 0-1, y 0-1 | 0x11, 2 |
| 8 | 0x00-0x13 | x 0-1, y 0-3 | 0x13, 4 |
| 16 | 0x00-0x43, the whole package | x 0, 1, 3, 4, y 0-3 | 0x43, 7 (through the memory column) |

## Results

Runs `runs/stg_sync_ab`, `runs/stg_sync_c`, `runs/stg_sync_d`, all PASS; every barrier
report err = 0 on every participant. Cycles of the 500 MHz cluster clock.

| P | hops | A barrier | B barrier | C 1-to-1 | C 1-to-all | D 1-to-1 | D 1-to-all |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 742 | 654 | 3019 | 3139 | 186 | 186 |
| 4 | 2 | 960 | 777 | 3472 | 4819 | 204 | 209 |
| 8 | 4 | 1382 | 992 | 4129 | 9589 | 240 | 255 |
| 16 | 7 | 2546 | 1632 | 5089 | 23451 | 292 | 292 |

- At P = 16, against the broadcast barrier: D is 5.6x cheaper, C's 1-to-all 14.4x dearer; C / D = 80x.
- D: the P = 2 identity check (1-to-1 and 1-to-all are the same graph) holds to the cycle (186 = 186);
  30 edges cost the same as 2 at P = 16 (292 = 292). Local control (both tasks on chip 0x00) 138 cc.
- C: P = 2 reads 3019 vs 3139, the systematic spread of the software manager's one-to-all graph. Local
  control 4178 cc, which tracks the graph's task count (57 here), so `rt - local` is not a network cost.
  L2 task heap peak 29,184 of 125,952 B, no OOM.
- The CSV's `min` / `max` for A and B are the extremes over all 32 iterations and all participants. B's
  spread is widest at P = 16: 1503 / 1632 / 2750 (min / avg / max).

