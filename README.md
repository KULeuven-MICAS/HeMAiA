![CI](https://github.com/KULeuven-MICAS/HeMAiA/actions/workflows/ci.yml/badge.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# HeMAiA

This repository hosts the hardware and software for the HeMAiA system-on-chip and its generator. HeMAiA is a scalable SoC that derived from [Occamy](https://github.com/pulp-platform/occamy/) and is customized to be able to integrate multiple heterogeneous [SNAX Cluster](https://github.com/KULeuven-MICAS/snax_cluster). The SoC has two buses: 64-bit bus for processors, and 512-bit bus for DMA data copy between different clusters. 

## License

Occamy and HeMAiA are being made available under permissive open source licenses.

The following files are released under Apache License 2.0 (`Apache-2.0`) see `LICENSE`:

- `sw/`
- `util/`

The following files are released under Solderpad v0.51 (`SHL-0.51`) see `hw/LICENSE`:

- `hw/`

## Getting Started

To get started, you need to use the following command to pull the correct version of SNAX Docker image and mount your home directory to the container: 

```bash
docker run --workdir $(realpath ~) -it -v $(realpath ~):$(realpath ~) ghcr.io/kuleuven-micas/hemaia:main
```

Then you can go to the folder where the HeMAiA repo is located at, and execute the following three commands in the container. These four commands are necessary for either simulation or FPGA prototyping:

- Compile the software. The default CFG is `target/rtl/cfg/hemaia_ci.hjson`. You can override the configuration to the one you want:

```bash
make sw CFG_OVERRIDE=target/rtl/cfg/<YOUR_CFG>
```

Alternatively, you can compile a specific software workload with more precise control using `make single-sw`:

```bash
make single-sw HOST_APP_TYPE=<TYPE> CHIP_TYPE=<CHIP> WORKLOAD=<WORKLOAD> DEV_APP=<DEV_APP>
```

The parameters for `single-sw` (and `apps`) are:
- `HOST_APP_TYPE`: The category of the host application (`host_only`, `offload_legacy`, `offload_bingo_sw`, `offload_bingo_hw`).
- `CHIP_TYPE`: The target platform configuration (`single_chip`, `multi_chip`).
- `WORKLOAD`: The specific workload name (e.g., `gemm_tiled`).
- `DEV_APP`: The target device-side accelerator (e.g., `snax-bingo-offload`).

We have different software runtime modes:

- **Offload Legacy**: After initialization of the SoC, CVA6 transfers the function pointer to a Snitch core and waits for an interrupt. After execution, CVA6 collects the returned values.

- **Bingo SW Runtime**: A software-managed runtime where the CVA6 schedules workloads represented as Directed Acyclic Graphs (DAGs). Scheduling operates at the **cluster level**, meaning each node in the DFG corresponds to a cluster-wide task. The kernel handles intra-cluster synchronization, while the runtime manages global synchronization.

- **Bingo HW Runtime**: A high-performance runtime that offloads scheduling to the dedicated **Bingo Hardware Manager**. A hardware scheduler manages ready/done queues, allowing cores to poll for tasks via specialized CSRs with zero-overhead dispatch. This enables **core-level scheduling** with fine task granularity. Both cluster-level and global-level synchronization are handled automatically by the hardware manager.

- Convert one compiled elf file to the hex file so that it can be preloaded into SRAM:

```bash
make apps HOST_APP_TYPE=<TYPE> CHIP_TYPE=<CHIP> WORKLOAD=<WORKLOAD> DEV_APP=<DEV_APP>
```

- Compile the Bootrom:

```bash
make bootrom
```

- Compile the SystemVerilog Code:

```bash
make rtl CFG_OVERRIDE=target/rtl/cfg/<YOUR_CFG>
```

### Perform RTL simulation using Verilator

Verilator is the open source SystemVerilog / Verilog simulator. Verilator is embedded in the docker image, thus there is no need to exit the container.

- Compile the Verilator binary:

```bash
make hemaia_system_vlt -j$(nproc)
```

- Execute the compiled binary:

```bash
# @ target/sim/bin
./occamy_chip.vlt
```

### Perform RTL simulation using Questasim

Questasim is the commercial SystemVerilog / Verilog simulator. The generation of Questasim start script will be done in the container, and actual compilation / execution of the simulation needs the environment with Questasim installation.

- Prepare the compilation script for Questasim:

```bash
make hemaia_system_vsim_preparation
```

- Exit the docker image and compile the simulation binary in the environment with Questasim installation:

```bash
make hemaia_system_vsim
```

- Execute the compiled binary:

```bash
# @ target/sim/bin
./occamy_chip.vsim[.gui]
```

### Perform RTL simulation using VCS

VCS is the commercial SystemVerilog / Verilog simulator. The generation of VCS start script will be done in the container, and actual compilation / execution of the simulation needs the environment with VCS installation.

- Prepare the compilation script for VCS:

```bash
make hemaia_system_vcs_preparation
```

- Exit the docker image and compile the simulation binary in the environment with VCS installation:

```bash
make hemaia_system_vcs
```

- Execute the compiled binary, with optional parameter to enable Verdi GUI and determine number of simulation threads:

```bash
# @ target/sim/bin
./occamy_chip.vcs [-gui] [-num_threads:8]
```

### Perform FPGA Prototype

HeMAiA now only provides the support for AMD Versal VPK180 board. Vivado 2023.2 is required.

- Prepare the compilation script for Vivado inside the container:

```bash
make hemaia_system_vivado_preparation CFG_OVERRIDE=target/rtl/cfg/<YOUR_CFG> TARGET_PLATFORM=vpk180
```

- Exit the container and call Vivado to synthesize and implement the hardware. The constraint is especially for the MICAS Debugger board, thus modification may be needed for different peripherals. The constraint is at **target/fpga_chip/hemaia_system/hemaia_system_vpk180_impl.xdc** and **target/fpga_chip/hemaia_system/hemaia_system_vpk180_impl_ext_jtag.xdc**.

```bash
make hemaia_system_vivado TARGET_PLATFORM=vpk180
```

- After implementation, open the Vivado and program bitstream to the board:

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

- Open minicom and execute the binary by pressing number 7.
