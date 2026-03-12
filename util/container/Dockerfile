# Author: Fanchen Kong <fanchen.kong@kuleuven.be>

# Docker container for HEMAIA

ARG UBUNTU_VERSION=24.04


#-------------------------------------------------------------------------------
# Stage 1: Tool builder
# Installs generic tool binaries.
#-------------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS tool-builder
# Install APT requirements
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    # General requirements
    curl wget tar nano python3 pip gcc zsh device-tree-compiler libboost-regex-dev libboost-system-dev autoconf automake autotools-dev python3-pip python3-tomli libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git cmake libglib2.0-dev libslirp-dev libncurses-dev
# Install Oh-My-Zsh and Autocomplete Plugin
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone https://github.com/zsh-users/zsh-autosuggestions.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    echo "source ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh" >> ~/.zshrc && \
    chsh -s $(which zsh)
# Zsh as default shell
CMD ["zsh"]

#-------------------------------------------------------------------------------
# Stage 2: LLVM Toolchain for Snitch Core
#-------------------------------------------------------------------------------
ARG SNITCH_LLVM_VERSION=15.0.0-snitch-0.2.0
RUN wget https://github.com/pulp-platform/llvm-project/releases/download/${SNITCH_LLVM_VERSION}/riscv32-snitch-llvm-ubuntu2204-${SNITCH_LLVM_VERSION}.tar.gz && \
    tar xvf riscv32-snitch-llvm-ubuntu2204-${SNITCH_LLVM_VERSION}.tar.gz && \
    mv riscv32-snitch-llvm-ubuntu2204-${SNITCH_LLVM_VERSION} riscv-llvm && \
    rm -rf riscv32-snitch-llvm-ubuntu2204-${SNITCH_LLVM_VERSION}.tar.gz
ENV PATH="/riscv-llvm/bin:${PATH}"

# Spike for traces
RUN wget https://github.com/KULeuven-MICAS/HeMAiA/releases/download/hemaia_toolchain/spike.tar.gz && \
    tar xvf spike.tar.gz && \
    mv spike /opt/spike && \
    rm -rf riscv.tar.gz
# Build from scratch (optional, as pre-built binaries are provided above)
# git clone https://github.com/riscv-software-src/riscv-isa-sim.git && \
#     cd riscv-isa-sim && \
#     ./configure --prefix=/opt/spike/ && \
#     make install -j && \
#     cd .. && \
#     rm -rf riscv-isa-sim
ENV PATH="/opt/spike/bin:${PATH}"

#-------------------------------------------------------------------------------
# Stage 3: GCC 64bit Toolchain for Host CVA6 Core
#-------------------------------------------------------------------------------
RUN wget https://github.com/KULeuven-MICAS/HeMAiA/releases/download/hemaia_toolchain/riscv.tar.gz && \
    tar xvf riscv.tar.gz && \
    mv riscv /opt/riscv && \
    rm -rf riscv.tar.gz
# Build from scratch (optional, as pre-built binaries are provided above)
# see https://github.com/riscv-collab/riscv-gnu-toolchain
# git clone https://github.com/riscv/riscv-gnu-toolchain.git && \
#     cd riscv-gnu-toolchain && \
#     ./configure -prefix=/opt/riscv --with-arch=rv64gcv --with-abi=lp64d && \
#     make -j && \
#     cd .. && \
#     rm -rf riscv-gnu-toolchain
ENV PATH="/opt/riscv/bin:${PATH}"

#-------------------------------------------------------------------------------
# Stage 4: Verilator
#-------------------------------------------------------------------------------
# Follow the https://verilator.org/guide/latest/install.html
ARG VERILATOR_VERSION=5.024
RUN apt-get install -y git help2man perl python3 make autoconf g++ flex bison ccache \
    libgoogle-perftools-dev numactl perl-doc \
    libfl2 \
    libfl-dev && \
    git clone https://github.com/verilator/verilator && \
    unset VERILATOR_ROOT && \
    cd verilator && \
    git pull && \
    git checkout v${VERILATOR_VERSION} && \
    autoconf && \
    ./configure --prefix=/opt/verilator && \
    make -j && \
    make install
ENV PATH="/opt/verilator/share/verilator/bin:${PATH}"
#-------------------------------------------------------------------------------
# Stage 5: Chisel Env
#-------------------------------------------------------------------------------
# First the jdk
RUN apt install -y wget gpg apt-transport-https && \
    wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null && \
    echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list && \
    apt update && \
    apt install -y temurin-17-jdk
# Then the sbt
RUN curl -s -L https://github.com/sbt/sbt/releases/download/v1.9.7/sbt-1.9.7.tgz | tar xvz && \
    mkdir -p /tools/sbt && \
    mv sbt/bin/sbt /tools/sbt
ENV PATH="/tools/sbt:${PATH}"

#-------------------------------------------------------------------------------
# Stage 5: Python Deps and other tools
#-------------------------------------------------------------------------------
# Install Python dependencies for HEMAIA
RUN apt-get install -y python3-pip && \
    python3 -m pip config set global.break-system-packages true && \
    pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    pandas \
    hjson \
    mako \
    jsonref \
    jsonschema \
    tabulate \
    pyyaml \
    networkx
# Bender
RUN wget https://github.com/KULeuven-MICAS/HeMAiA/releases/download/hemaia_toolchain/bender && \
    chmod +x bender && \
    mkdir -p /tools && \
    mv bender /tools
# Get Bender (optional, as pre-built binary is provided above)
# WORKDIR /tools
# RUN curl --proto '=https' --tlsv1.2 https://pulp-platform.github.io/bender/init -sSf | sh -s
ENV PATH="/tools:${PATH}"