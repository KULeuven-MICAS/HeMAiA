# E3 synchronisation: the cost of one cross-chip synchronisation, four mechanisms

One button: `./run.sh` re-plots `figs/fig_e3_four_arm.{pdf,png}` from `results/four_arm_sync_latency.csv`.
Nothing is simulated here. Per-edge synchronisation shows its worth in the intermediate and final
experiments, where it runs inside real workloads.

## The four mechanisms

| arm | mechanism | how a dependency is enforced |
|---|---|---|
| A | software barrier | point-to-point stores: the master collects arrivals and pushes the release |
| B | 0xFF broadcast barrier | one store, multicast by the D2D router |
| C | BINGO software manager | host-mediated: mailbox, then manager, then dispatch, per edge |
| D | BINGO hardware manager | the same graph, resolved in RTL |

A and B are barriers: everyone waits for everyone. C and D are per edge: a task waits only for the
producers it declares. A and B are measured as one barrier; C and D as a round trip (two edges), for a
1-to-1 and a 1-to-all edge.

## Data

`results/four_arm_sync_latency.csv` is copied from `dev_docs/hemaiaV2/four-arm-sync-latency.csv`,
measured on `hemaia_16chiplet.hjson` (4x4 grid, memory chiplet at [4,0], VCS). The method and the raw
numbers are in `dev_docs/hemaiaV2/07-four-arm-sync-comparison.md`.

| P | A barrier | B barrier | C 1-to-1 | C 1-to-all | D 1-to-1 | D 1-to-all |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 976 | 870 | 5313 | 5600 | 164 | 164 |
| 4 | 1084 | 987 | 6133 | 8470 | 173 | 184 |
| 8 | 1394 | 1203 | 7609 | 17900 | 191 | 204 |
| 16 | 2025 | 1653 | 9618 | 45370 | 210 | 209 |

Cycles of the 500 MHz cluster clock.
