# VPK180 DRAM

The VPK180 recipes enable the DRAM bridge when the generated SoC has an
`external_memory` port. `hemaia_cva6_only.hjson` and `hemaia_vpk180.hjson`
provide a 12 GiB aperture. The chip ID selects the local aperture. East and
west use their existing VIO chip ID straps. The single-board recipe builds
with all four physical D2D interfaces disabled. Set the strap while the SoC is in
reset. Quiesce DDR accesses before resetting the SoC through VIO or the reset
button. The hard NoC and its DDR controllers retain their state across a SoC
reset.

| HeMAiA byte address | NoC base | Capacity | VPK180 triplet |
| --- | --- | --- | --- |
| `0x??20_0000_0000` | `0x0000_0000_0000` | 2 GiB | 3 |
| `0x??20_8000_0000` | `0x0008_0000_0000` | 2 GiB | 3 |
| `0x??21_0000_0000` | `0x0500_0000_0000` | 4 GiB | 2 |
| `0x??22_0000_0000` | `0x0600_0000_0000` | 4 GiB | 1 |

The address mapper feeds the native 512-bit PL NoC port on `clk_host`.
The hard NMU handles clock-domain crossing and rate matching to the 960 MHz
NoC. Vivado models the host interface at 40 MHz, the timing bound used by
the FPGA recipes. Let outstanding DRAM requests finish before reprogramming
or stopping the host clock. The DDRMC board presets set pinout, DRAM timing
and automatic refresh.
Each DDRMC stripes consecutive 256-byte blocks across its two LPDDR4 channels.

Generate RTL and the FPGA boot ROM in the HeMAiA container. On the host, use
Vivado 2025.2 to build the desired systems.

```sh
# In the HeMAiA container, from the repository root
make bootrom rtl CFG_OVERRIDE=target/rtl/cfg/hemaia_cva6_only.hjson

# On the host, from the repository root
make hemaia_system_vivado TARGET_PLATFORM=vpk180 DEBUG=0 EXT_JTAG=1
make hemaia_system_east_vivado TARGET_PLATFORM=vpk180 DEBUG=0 EXT_JTAG=1
make hemaia_system_west_vivado TARGET_PLATFORM=vpk180 DEBUG=0 EXT_JTAG=1
```

Set `HEMAIA_VIVADO_IP_CACHE` to share a Vivado IP cache across the three builds.

Run the standalone address mapper test with Verilator.

```sh
make -C target/fpga/common/test
```
