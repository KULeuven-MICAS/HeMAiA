![CI](https://github.com/KULeuven-MICAS/HeMAiA/actions/workflows/ci.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# HeMAiA

This repository hosts the hardware and software for the HeMAiA system-on-chip and its generator. HeMAiA is a scalable SoC that derived from [Occamy](https://github.com/pulp-platform/occamy/) and is customized to be able to integrate multiple heterogeneous [SNAX Cluster](https://github.com/KULeuven-MICAS/snax_cluster). The SoC has two buses: 64-bit bus for processors, and 512-bit bus for DMA data copy between different clusters. 


## Getting Started
To get started, you need to use the following command to pull the correct version of SNAX Docker image and mount your home directory to the container: 
```bash
docker run --workdir $(realpath ~) -it -v $(realpath ~):$(realpath ~) ghcr.io/kuleuven-micas/snax@sha256:4ff37cad4e85d6a898cda3232ee04a1210833eb4618d1f1fd183201c03c4c57c
```
Then you can go to the folder where the HeMAiA repo is located at, and execute the following three commands in the container. These three commands are necessary for either simulation or FPGA prototyping: 

- Compile the Bootrom: 
```bash
make bootrom CFG_OVERRIDE=target/rtl/cfg/...
```
- Compile the SystemVerilog Code: 
```bash
make rtl CFG_OVERRIDE=target/rtl/cfg/...
```
- Compile the software: 
```bash
make sw CFG_OVERRIDE=target/rtl/cfg/... -j24
```

### Perform RTL simulation using Verilator:
- Compile the Verilator binary:
```bash
make occamy_system_vlt -j24
```
- Execute the compiled binary:
```bash
# @ target/sim
cd target/sim;bin/occamy_top.vlt sw/host/apps/offload/build/[choose any ELF]
```

### Perform RTL simulation using Questasim:
- Prepare the compilation script for Questasim:
```bash
make occamy_system_vsim_preparation
```
- Exit the docker image as there is no Questasim in the container. Compile the Questasim binary:
```bash
make occamy_system_vsim
```
- Execute the compiled binary:
```bash
cd target/sim;bin/occamy_top.vsim[.gui] sw/host/apps/offload/build/[choose any ELF]
```

### Perform FPGA Prototype:
- Extract the binary from **elf** file:
```bash
cd target/fpga_chip/apps;make apps
```
- Prepare the compilation script for Vivado:
```bash
make hemaia_system_vivado_preparation CFG_OVERRIDE=target/rtl/cfg/... TARGET_PLATFORM=vpk180
```
- Exit the docker image as there is no Vivado in the container. Call Vivado to synthesize and implement the hardware. The constraint is especially for the MICAS Debugger board, thus modification may be needed for different peripherals. The constraint is at **target/fpga_chip/hemaia_system/hemaia_system_vpk180_impl.xdc** and **target/fpga_chip/hemaia_system/hemaia_system_vpk180_impl_ext_jtag.xdc**. 
```bash
make hemaia_system_vivado TARGET_PLATFORM=vpk180
```

- Open the Vivado and perform the hardware programming and reset the chip:
```bash
make hemaia_system_vivado_gui
```
After programming the hardware, adds the virtual button with default value of 0 in ***VIO*** tab. Press the virtual button to reset the hardware. 

- Open minicom to interact with the FPGA:
```bash
minicom -D /dev/ttyUSB? -b 500000
```
If you open the correct UART, then you will see Bootrom printing something. 
After finding the correct port, exit minicom by pressing **Ctrl + A**, then **X**. 

- Modify the Software download script to the correct port. The file is at ***target/fpga_chip/apps/send_uart.sh***. Every UART string should be replaced by the one that you found earlier. 

- If you see the **Welcome to HeMAiA Bootrom** from minicom (after pressing one random key on the keyboard), you are ready to download the binary to the SoC. You can download the binary by exeucting: 
```bash
cd target/fpga_chip_apps
./send_uart.sh [Followed by the path of the binary]
```
It takes some time for the handshaking between the computer and the SoC (20 seconds). At the end you will see transfer complete. 

- Open minicom and execute the binary by pressing 4. 

## How to Build Your Own Program

The relevant directory contents for making your own program are listed below:

- `target/sim/sw/device/apps/snax`: contains all the snax programs.
  - In each program you will see the necessaryt source and data files.
- `target/sim/sw/device/snax`: contains all the pre-built kernel libraries. These are sourced by the programs made in the `apps` directory.
- `target/sim/sw/device/host`: contains the programs for the CVA6 to run.
- `target/sim/sw/device/host/apps/offload`: contains the most relevant source code for the CVA6. The source C file simply initializes the host and snax cores and waits for them to finish. Take note that snax cores are like "devices" in this regard.
- `target/sim/sw/device/host/apps/offload/build`: contains all the compiled `.elf` files to be run by the host. Since HEMAIA is a host-system, all programs start running from the host.

Programs are written by:
- Follow from several examples inside the `target/sim/sw/device/apps/snax` directory.
  - This is similar to programming a simple cluster shell from SNAX cluster.
  - Put all your starting `data.h` and source codes in this directory.
  - Take note, that you have to also specify in the program, which cluster core you are using. So there is an extra layer of conditional statement to indicate which cluster shell is running:

```C
    // n - is the cluster number you are using
    // e.g., in HEMAIA, cluster number 3 is the DIMC
    if (snrt_cluster_idx() == n){
        // Put your original SNAX cluster code in here
    }
```

- If you will use pre-built kernels for your accelerator, put them in: `target/sim/sw/device/snax`
- Modify `target/sim/sw/host/apps/offload/Makefile` and add to the `DEVICE_APPS` list the path to the new program you are using.

```Makefile
DEVICE_APPS += [insert the path to your program]
```

- When you accomplish all you can build your SW: `make CFG_OVERRIDE=... sw -j`

## Content

What can you expect to find in this repository?

- The HeMAiA SoC generator and hardware sources/templates.
- A baremetal runtime for HeMAiA.
- Two RTL simulation environments. One is the originally designed by Pulp and another is designed by us. The former one supports either Verilator and Questasim, and the later one supports VCS additionally. 

[SNAX Cluster](https://github.com/KULeuven-MICAS/snax_cluster) is hosted in a seperated repo, and can be automatically pulled by [Bender](https://github.com/pulp-platform/bender). 

## License

Occamy is being made available under permissive open source licenses.

The following files are released under Apache License 2.0 (`Apache-2.0`) see `LICENSE`:

- `sw/`
- `util/`

The following files are released under Solderpad v0.51 (`SHL-0.51`) see `hw/LICENSE`:

- `hw/`

# Getting start for HeMAiA full-chip simulation & prototyping

### **⚠️Tips**: All commands is made at the root dir of the HeMAiA repo. 

## Prototype the system on FPGA:
### Option 1: Occamy configuration on VCU128 (Bootrom and DRAM from Xilinx IP)

```makefile
@ SNAX Docker: make bootrom CFG_OVERRIDE=target/rtl/cfg/...
@ SNAX Docker: make sw
@ SNAX Docker: make -C target/fpga/sw [APP=???] (Which binary file you want to use)
@ SNAX Docker: make rtl CFG_OVERRIDE=target/rtl/cfg/...hjson
@ SNAX Docker: make occamy_system_vivado_preparation
@ Barnard3: make occamy_system_vcu128
@ Barnard3: make occamy_system_vcu128_gui
```

### Option 2: HeMAiA configuration on VPK180 (Tapeout configuration, everything is internal)

```makefile
@ SNAX Docker: make bootrom CFG_OVERRIDE=target/rtl/cfg/...hjson
@ SNAX Docker: make sw CFG_OVERRIDE=target/rtl/cfg/...hjson
@ SNAX Docker: make -C target/fpga/sw [APP=???] (Which binary file you want to use)
@ SNAX Docker: make rtl CFG_OVERRIDE=target/rtl/cfg/...hjson
@ SNAX Docker: make hemaia_system_vivado_preparation TARGET_PLATFORM=[vpk180|vcu128]
@ Barnard3: make hemaia_system_vivado TARGET_PLATFORM=[vpk180|vcu128]
@ Barnard3: make hemaia_system_vivado_gui
```

## Simulate the system: 
### Option 1: Simulate with Verilator

```makefile
@ SNAX Docker: make bootrom CFG_OVERRIDE=target/rtl/cfg/...hjson
@ SNAX Docker: make sw CFG_OVERRIDE=target/rtl/cfg/...hjson
@ SNAX Docker: make rtl CFG_OVERRIDE=target/rtl/cfg/...hjson
@ SNAX Docker: make occamy_system_vlt
@ SNAX Docker: cd target/sim/bin;./occamy_top.vlt sw/host/apps/offload/build/[choose any ELF] [--vcd]
```

### Option 2: Simulate with Questasim


