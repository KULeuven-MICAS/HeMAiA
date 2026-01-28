# Host-Only Applications

This directory contains host-only applications for the HeMAiA project. These applications run on the host processor without offloading to device accelerators. They are useful for system configuration, testing, and standalone utilities.

## Applications

- **hello_world**: A simple "Hello World" program to verify the host runtime and compilation setup.
- **hemaia_clk_rst_configurator**: Configures clock and reset settings for the HeMAiA system.
- **hemaia_d2d_link_configurator**: Configures die-to-die (D2D) links for multi-chip communication.
- **hemaia_multichip_mailbox**: Demonstrates mailbox-based communication between chiplets.
- **hemaia_multichip_sync**: Handles synchronization across multiple chips in a multi-chip setup.
- **hemaia_system_xdma_copy**: Performs XDMA-based data copying operations for system-level data transfer.

## Building Applications

To build all host-only applications:
```
make host_only
```

To build a specific application (e.g., hello_world):
```
make app APP=hello_world
```

Or navigate to the app directory and run:
```
make -C apps/host_only/hello_world
```

## Adding a New Host-Only Application

Follow these steps to add a new application under `host_only`:

1. **Create the Application Directory**:
   Create a new directory under `apps/host_only/`, e.g., `new_app/`.

2. **Add Source Files**:
   - Create a `src/` subdirectory.
   - Place your C source files in `src/`, e.g., `new_app.c`.

3. **Create the Makefile**:
   Create `Makefile` in the new app directory with the following content:
   ```
   APP = new_app
   SRCS = src/new_app.c  # Add more sources if needed
   BINGO_HOST ?= 0       # Set to 1 if using Bingo runtime
   INCL_DEVICE_BINARY ?= 0  # Set to 1 if including device binaries
   include ../../common.mk
   ```
   - `APP`: The name of the application (must match the directory name).
   - `SRCS`: List of source files.
   - `BINGO_HOST`: Enable if the app uses the Bingo runtime library. 
   - `INCL_DEVICE_BINARY`: Enable if the app needs to link device binaries. For the host_only application, set to 0.

4. **Update the Top-Level Makefile**:
   Add the new app to `HOST_ONLY_APPS` in `/target/sw/host/Makefile`:
   ```
   HOST_ONLY_APPS += new_app
   ```

5. **Build and Test**:
   - Run `make app APP=new_app` to build.
   - The output will be in `build/` (e.g., `new_app.elf`, `new_app.bin`).
   - Test by running the generated binary on the target hardware.

For more details on build options, refer to `common.mk`.