This workload uses the real TPS6287x I2C driver for DVFS. Each of the four
compute chiplets initializes a PMIC on its own local I2C bus before waking its
clusters. It then services the hardware manager's idle/busy requests.

The test starts at 800 mV and division 7 (about 571 MHz), raises to 1000 mV
and division 5 (800 MHz) on a busy request, and returns to 800 mV /7 after
the workload. D2D keeps the topology initializer's original clock settings:
4 GHz in the same-speed configuration, with the existing 200 MHz FPGA-link
exception otherwise. No extra clock programming is performed on chip20.
The D2D digital receiver shares the host clock; keeping the host at division
7 or faster preserves its required throughput relative to the 4 GHz PHY.

Edit these `#define` parameters directly in `target/sw/host/runtime/dvfs.h`
for the board. They are not Make variables; editing the header triggers a rebuild.

| Definition | Meaning |
| --- | --- |
| `DVFS_PMIC_ADDR` | Unshifted 7-bit PMIC address, `0x40` through `0x43`; default `0x41`. |
| `DVFS_PERIPH_HZ` | Actual I2C peripheral clock in Hz, independent of CPU DVFS. |
| `DVFS_NORMAL_MV` | Normal operating voltage, 400–1200 mV. |
| `DVFS_IDLE_MV` | Idle voltage, 400 mV through the normal voltage. |
| `DVFS_SETTLE_US` | Board-validated settling interval during initialization and RAISE. |
| `DVFS_MASTER_HZ` | Master/PLL clock before division; defaults to 4 GHz. |

The initial values (`0x41`, 32 MHz, 1000/800 mV, 50 us) are simulation
examples, not validated board settings. Before running, verify the address,
peripheral clock and guard interval, and choose voltages appropriate for the
SoC's frequency/voltage characterization. The PM
levels are clock divisors (`5` normal, `7` idle), defined in `bingo_api.h` and
used by both the scheduler and DVFS. They are not millivolt values. Voltage
requests are rounded to the nearest 5 mV by `tps6287x_set_voltage()`.

After editing the definitions, build from the repository root in the HeMAiA
container:

```sh
make single-sw \
  CFG_OVERRIDE=target/rtl/cfg/hemaia_tapeout_1c.hjson \
  HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=multi_chip \
  WORKLOAD=dvfs_cross_trigger_4chiplet_1cluster DEV_APP=snax-bingo-offload
```

Initialization calls `tps6287x_init()`, selects the idle divider, sets the idle
voltage, waits, and seeds ACK with the idle level before arming the interrupt
handler. A RAISE request sets voltage and waits before increasing
frequency; a LOWER request programs the lower frequency before sending the I2C
voltage write. `dvfs_wait_settle()` snapshots the CPU `mcycle` CSR at entry, reads
channel 0's divider once, and waits for
`ceil(DVFS_MASTER_HZ * settle_us / (divider * 1000000))` elapsed CPU cycles.
Unsigned subtraction handles counter rollover. Cycle counters must be enabled,
and no other software may change the PLL/dividers during the transition.

LOWER acknowledges after the voltage write succeeds; the already lowered clock
is safe while voltage ramps down. A following RAISE always waits after its own
voltage write. The ISR performs no UART output; it stores a bounded log for
`dvfs_dump_log()` to print after the workload. It clears the doorbell before
writing ACK, preserving any new notification caused by that ACK.

After the workload stops, the test disables automatic PM requests, applies
LOWER through the same DVFS transition routine, and records the idle ACK.

`DVFS_ACK` is written only after the sequence completes. An I2C error or an
unrecognized PM level aborts the test without acknowledging the failed request.
`CLOCK_VALID` clearing is not interpreted as a divider-applied acknowledgement.
Clock changes have no explicit delay. At 4 GHz the 8-bit divider/CDC latency is
well below 1 us; the subsequent address/register/data I2C write at <=1 MHz
takes tens of microseconds before the PMIC applies the new voltage. This covers
the divider transition on LOWER and during initialization. Before RAISE the
voltage has already settled.
The voltage settling interval must cover the board's analog transition time;
the PMIC does not report a dedicated DVS-done flag.

The testharness connects one `target/sim/testharness/pmic/tps6287x.sv` model
at address `0x41` to each compute chiplet, with external SDA/SCL pullups. The
model implements the datasheet single-register I2C protocol, register defaults
and nominal DVS ramp. It does not simulate a switching power stage or load.

Use `sim_rtl_with_pll.hjson` with these defaults: its PLL is 4 GHz and its
peripheral clock is 32 MHz. Prepare inside the HeMAiA container, from repo root:

```sh
make rtl CFG_OVERRIDE=target/rtl/cfg/hemaia_tapeout_1c.hjson BENDER="bender -d $PWD"
make -C target/rtl/bootrom/bootrom_sim bootrom
make apps CFG_OVERRIDE=target/rtl/cfg/hemaia_tapeout_1c.hjson \
  HOST_APP_TYPE=offload_bingo_hw CHIP_TYPE=multi_chip \
  WORKLOAD=dvfs_cross_trigger_4chiplet_1cluster DEV_APP=snax-bingo-offload \
  BENDER="bender -d $PWD"
make hemaia_system_vcs_preparation SIM_WITH_WAVEFORM=0 \
  SIM_CFG="$PWD/target/sim/cfg/sim_rtl_with_pll.hjson" BENDER="bender -d $PWD"
```

The Bender root argument makes nested SNAX generators use the SoC's resolved
dependencies. On the host with VCS installed, from repo root:

```sh
make hemaia_system_vcs PREPARED_SIM_INPUTS=1 SIM_WITH_WAVEFORM=0 \
  SIM_CFG="$PWD/target/sim/cfg/sim_rtl_with_pll.hjson" \
  VCS_TESTHARNESS_PREREQUISITES="$PWD/target/sim/testharness/testharness.sv"
cd target/sim/bin
./occamy_chip.vcs +check_pmic_dvfs
```

`+check_pmic_dvfs` checks applied host/cluster dividers against the modeled
voltage and requires successful PMIC initialization and DVFS writes on every
chiplet. This checker uses the workload's 1000/800 mV and 5/7 divider settings;
update its thresholds if those settings change. Success requires all four
chiplets to complete the 800 -> 1000 -> 800 mV sequence, finish at division 7,
retain their original D2D TX dividers, and pass the workload and PMIC checks.

The native driver and sequencing tests are `util/tests/test_tps6287x.py` and
`util/tests/test_dvfs.py`. The independent SystemVerilog bus test is
`util/tests/tps6287x_tb.sv` (top `tps6287x_tb`), compiled with the PMIC model.
