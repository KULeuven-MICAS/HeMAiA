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
docker run --workdir $(realpath ~) -it -v $(realpath ~):$(realpath ~) ghcr.io/kuleuven-micas/snax@sha256:4ff37cad4e85d6a898cda3232ee04a1210833eb4618d1f1fd183201c03c4c57c
```
Then you can go to the folder where the HeMAiA repo is located at, and execute the following three commands in the container. These four commands are necessary for either simulation or FPGA prototyping: 

- Compile the software: 
```bash
make sw -j$(nproc)
```
The default CFG is the target/rtl/cfg/hemaia_ci.hjson. You can override the configuration by 
```bash
make sw -j$(nproc) CFG_OVERRIDE=target/rtl/cfg/<YOUR_CFG>
```
- Compile the App:
 ```bash
make apps
```
The default app is specified in HOST_APP and DEVICE_APP. You can override this configuraiton by 
 ```bash
make apps HOST_APP=<YOUR_HOST_APP_NAME> DEVICE_APP=<YOUR_DEVICE_APP_NAME>
```

- Compile the Bootrom: 
```bash
make bootrom
```
- Compile the SystemVerilog Code: 
```bash
make rtl
```

### Perform RTL simulation using Verilator:
- Compile the Verilator binary:
```bash
make hemaia_system_vlt -j$(nproc)
```
- Execute the compiled binary:
```bash
# @ target/sim_chip
bin/occamy_chip.vlt
```

### Perform RTL simulation using Questasim:
- Prepare the compilation script for Questasim:
```bash
make hemaia_system_vsim_preparation
```
- Copy the software to be preloaded into SRAM:
```bash
# @ target/sim_chip/apps
python3 copy_m_n_times.py -i [ProgramName]
```
- Exit the docker image as there is no Questasim in the container. Compile the Questasim binary:
```bash
make hemaia_system_vsim
```
- Execute the compiled binary:
```bash
# @ target/sim_chip
bin/occamy_chip.vsim[.gui]
```

### Perform RTL simulation using VCS:
- Prepare the compilation script for VCS:
```bash
make hemaia_system_vcs_preparation
```

- Exit the docker image as there is no VCS in the container. Compile the VCS binary:
```bash
make hemaia_system_vcs
```

- Execute the compiled binary, with optional parameter to determine number of simulation threads:
```bash
# @ target/sim_chip
bin/occamy_chip.vcs [-gui] [-num_threads:8]
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
