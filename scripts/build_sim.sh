#!/bin/bash
# Bundle makefile commands to build RTL, software and simulator.
# Supports running in and outside of the SNAX container.

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bender update --fetch

podman run --rm -i -v "$(pwd)":"$(pwd)" -w "$(pwd)" ghcr.io/kuleuven-micas/snax:main bash -s <<'IN_CONTAINER'
set -e

make rtl
make bootrom
make sw
make occamy_system_vsim_preparation
IN_CONTAINER

# Make simulator
make occamy_system_vsim



