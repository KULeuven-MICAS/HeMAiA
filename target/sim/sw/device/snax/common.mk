MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(MK_DIR)/../toolchain.mk

###############
# Directories #
###############

# Fixed paths in repository tree
SNITCH_ROOT = $(shell bender path snitch_cluster)
RUNTIME_DIR = $(abspath $(MK_DIR)/../runtime)
SNRT_DIR    = $(SNITCH_ROOT)/sw/snRuntime
SW_DIR      = $(abspath $(MK_DIR)/../..)
MATH_DIR    = $(abspath $(MK_DIR)/../math)

# Paths relative to the app including this Makefile
BUILDDIR    = $(abspath build)
SRC_DIR     = $(abspath src)

###################
# Build variables #
###################

# Dependencies
INCDIRS += $(RUNTIME_DIR)/src
INCDIRS += $(SNRT_DIR)/api
INCDIRS += $(SNRT_DIR)/src
INCDIRS += $(SNRT_DIR)/vendor/riscv-opcodes
INCDIRS += $(SW_DIR)/shared/platform/generated
INCDIRS += $(SW_DIR)/shared/runtime
INCDIRS += $(SNRT_DIR)/../math/arch/riscv64/
INCDIRS += $(SNRT_DIR)/../math/arch/generic
INCDIRS += $(SNRT_DIR)/../math/src/include
INCDIRS += $(SNRT_DIR)/../math/src/internal
INCDIRS += $(SNRT_DIR)/../math/include/bits
INCDIRS += $(SNRT_DIR)/../math/include

SNRT_LIB_DIR  = $(abspath $(RUNTIME_DIR)/build/)
SNRT_LIB_NAME = snRuntime
SNRT_LIB      = $(realpath $(SNRT_LIB_DIR)/lib$(SNRT_LIB_NAME).a)

$(BUILDDIR):
	mkdir -p $@

$(BUILDDIR)/%.o: $(SRC_DIR)/%.c | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) $(RISCV_LDFLAGS) -c $< -o $@