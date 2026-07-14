# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# A recipe that dies after the shell has created/truncated `$@` leaves a partial
# file with a fresh mtime, which make then treats as up to date -- the next build
# would compile a half-written data header or .d.
.DELETE_ON_ERROR:

# 57 workload Makefiles declare `$(DATA_H) $(OFFLOAD_H) [$(CONFIGS_JSON)]: ...`. That is a
# NON-grouped multi-target rule: GNU make treats it as N independent rules that EACH run the
# whole recipe. Under -j make would launch main_bingo.py 2-3x concurrently, all writing the
# same files. We take our parallelism ACROSS apps (~96 of them), so serialising the inside of
# ONE app costs nothing and removes the hazard without editing 57 files.
# (Proper fix: grouped targets `&:`, GNU make >= 4.3.)
.NOTPARALLEL:

# Root directory (absolute path to the host directory)
ifndef HOST_DIR
HOST_DIR = $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
endif

# There are three main stages of building a host application:
# 1) host-partial-build: compile and link the host application without the device binary. This is mainly to determine the relocation
#    address of the device binary, which is needed to create the origin.ld file.
# 2) device-build: based on the generated origin.ld file, build the device application to get the device binary.
# 3) host-finalize-build: compile and link the host application with the device binary

# Each host application will include this common makefile to define the build rules.

# APP name composition
ifndef APP
    ifeq ($(HOST_APP_TYPE), host_only)
        APP = $(WORKLOAD)
    else ifeq ($(HOST_APP_TYPE), offload_legacy)
        APP = offload_legacy_$(CHIP_TYPE)
    else ifneq ($(HOST_APP_TYPE),)
        APP = $(HOST_APP_TYPE)_$(CHIP_TYPE)_$(WORKLOAD)
    endif
endif

# Validation of input variables
ALLOWED_HOST_APP_TYPES = host_only offload_legacy offload_bingo_sw offload_bingo_hw
ALLOWED_CHIP_TYPES     = single_chip multi_chip

# Skip validation for clean target
ifneq ($(MAKECMDGOALS),clean)
    ifeq ($(filter $(HOST_APP_TYPE),$(ALLOWED_HOST_APP_TYPES)),)
        $(error Invalid or missing HOST_APP_TYPE: "$(HOST_APP_TYPE)". Allowed values: $(ALLOWED_HOST_APP_TYPES))
    endif

    ifeq ($(filter $(CHIP_TYPE),$(ALLOWED_CHIP_TYPES)),)
        $(error Invalid or missing CHIP_TYPE: "$(CHIP_TYPE)". Allowed values: $(ALLOWED_CHIP_TYPES))
    endif

    # Check for WORKLOAD if not offload_legacy
    ifneq ($(HOST_APP_TYPE), offload_legacy)
        ifeq ($(WORKLOAD),)
            $(error WORKLOAD must be specified for HOST_APP_TYPE=$(HOST_APP_TYPE))
        endif
    endif

endif

# Resolve DEVICE_APPS -- the device binaries this host app embeds.
#
# There are two ways an app declares them, and they must not fight:
#   * a workload Makefile sets DEV_APP (e.g. `DEV_APP = snax-bingo-offload`) and
#     leaves DEVICE_APPS empty -- DEV_APP seeds the list;
#   * an app TREE (offload_legacy/*) sets DEVICE_APPS directly to a list of
#     several device apps, and sets no DEV_APP.
#
# `single-sw` narrows the build to one device app by passing DEV_APP on the
# COMMAND LINE, and only that origin may override a declared DEVICE_APPS list --
# a DEV_APP that a tree's own Makefile sets must not truncate the list that tree
# declares (offload_legacy/single_chip declares five device apps).
ifeq ($(origin DEV_APP),command line)
    DEVICE_APPS := $(DEV_APP)
else ifeq ($(strip $(DEVICE_APPS)),)
    DEVICE_APPS := $(DEV_APP)
endif

# Validate the thing that actually matters -- the resolved list, not DEV_APP: an app
# tree (offload_legacy/multi_chip) may declare DEVICE_APPS directly and set no DEV_APP.
ifneq ($(MAKECMDGOALS),clean)
    ifneq ($(filter $(HOST_APP_TYPE),offload_legacy offload_bingo_sw offload_bingo_hw),)
        ifeq ($(strip $(DEVICE_APPS)),)
            $(error No device app for $(HOST_APP_TYPE): set DEVICE_APPS or DEV_APP)
        endif
    endif
endif

######################
# Invocation options #
######################

DEBUG ?= OFF # ON to turn on debugging symbols

###################
# Build variables #
###################

# Compiler toolchain
# CVA6_GCC_ROOT = /opt/riscv
CVA6_GCC_ROOT ?= 
RISCV_CC      = $(CVA6_GCC_ROOT)riscv64-unknown-elf-gcc
RISCV_OBJCOPY = $(CVA6_GCC_ROOT)riscv64-unknown-elf-objcopy
RISCV_OBJDUMP = $(CVA6_GCC_ROOT)riscv64-unknown-elf-objdump
RISCV_READELF = $(CVA6_GCC_ROOT)riscv64-unknown-elf-readelf
# Directories
# Notice the build dir is the dir under each app
BUILDDIR    ?= $(abspath build)
# Global DIRs
SWDIR       = $(abspath $(HOST_DIR)/..)
RUNTIME_DIR = $(abspath $(HOST_DIR)/runtime)
DEVICE_DIR  = $(abspath $(HOST_DIR)/../device)

# Dependencies
INCDIRS += $(RUNTIME_DIR)
INCDIRS += $(abspath $(SWDIR)/shared/platform/generated)
INCDIRS += $(abspath $(SWDIR)/shared/runtime)
SRCS    += $(RUNTIME_DIR)/start.S

# Include XDMA
INCDIRS += $(SWDIR)/shared/vendor/xdma

# libbingo headers are always needed because host_kernel_lib.h (included via
# host.h in every host app) references libbingo symbols (BINGO_CHECK_TYPE_*,
# BINGO_GATING_MODE_*, BINGO_PRINTF, bingo_cerf_update, bingo_utils.h helpers).
# Linking against libbingo remains conditional on BINGO_HOST below.
INCDIRS += $(abspath $(HOST_DIR)/runtime/libbingo/include)
INCDIRS += $(SWDIR)/shared/vendor/bingo_alloc/bingo_alloc

# VersaCore shared headers — notably gemm_shapes.h, used by host workloads
# that size GEMM buffers from bingo_gemm_shape_params[array_shape]. The
# device-only snax_versacore_lib.h also lives here; host code should only
# include <gemm_shapes.h>.
INCDIRS += $(abspath $(DEVICE_DIR)/runtime/snax/versacore)

# Compiler flags
RISCV_CFLAGS += $(addprefix -I,$(INCDIRS))
RISCV_CFLAGS += -march=rv64gcv_zvfh # v extension for ara + zvfh for fp16 vectors
RISCV_CFLAGS += -mabi=lp64d
RISCV_CFLAGS += -mcmodel=medany
RISCV_CFLAGS += -ffast-math
RISCV_CFLAGS += -fno-builtin-printf
RISCV_CFLAGS += -fno-common
RISCV_CFLAGS += -O2
RISCV_CFLAGS += -fno-tree-vectorize -fno-tree-loop-vectorize -fno-tree-slp-vectorize
RISCV_CFLAGS += -ffunction-sections
RISCV_CFLAGS += -fdata-sections
RISCV_CFLAGS += -Wextra
RISCV_CFLAGS += -Werror
RISCV_CFLAGS += -std=gnu99
ifeq ($(DEBUG),ON)
RISCV_CFLAGS += -g
endif
RISCV_CFLAGS += $(USER_FLAGS)
# Linking sources
LINKER_SCRIPT = $(abspath $(HOST_DIR)/runtime/host.ld)
LD_SRCS       = $(LINKER_SCRIPT)
# Linker flags
RISCV_LDFLAGS += -nostartfiles
RISCV_LDFLAGS += -lm
RISCV_LDFLAGS += -lgcc
RISCV_LDFLAGS += -T$(LINKER_SCRIPT)
# The host runs bare-metal from a unified SPM (host.ld maps code+data+bss into
# one WIDE_SPM region), so the single PT_LOAD segment is intentionally RWX.
# Silence binutils >= 2.39's W^X advisory warning (the flag was introduced in
# the same release that added the warning, so any ld that warns accepts it).
RISCV_LDFLAGS += -Wl,--no-warn-rwx-segments
# GC unreferenced functions/data (paired with -ffunction-sections/-fdata-sections): the multichip
# 1-cluster chips have a small 128 KiB wide SPM, and linking all of libbingo's host kernels leaves
# too little for the L3 task-metadata heap -> host L3 OOM. Stripping unused kernels shrinks .text and
# frees the heap. The offload_bingo_hw.h references its live kernels by &symbol, so they are kept.
RISCV_LDFLAGS += -Wl,--gc-sections


# if the host application uses the bingo runtime
ifneq (,$(filter $(BINGO_HOST),True true TRUE 1))
LIBBINGO_DIR = $(abspath $(RUNTIME_DIR)/libbingo)


# libbingo (includes already added above unconditionally)
BINGO_LIB_DIR = $(LIBBINGO_DIR)/build
BINGO_LIB_NAME = bingo
BINGO_LIB = $(BINGO_LIB_DIR)/lib$(BINGO_LIB_NAME).a

LD_SRCS += $(BINGO_LIB) 

# link bingo lib
RISCV_LDFLAGS += -L$(BINGO_LIB_DIR)
RISCV_LDFLAGS += -l$(BINGO_LIB_NAME)

endif


###########
# Outputs #
###########

# Whatever the APP declared as generated (data headers, offload headers, ...) -- captured
# HERE, before common.mk appends its own outputs to PARTIAL_OUTPUTS below, so this is the
# app's generated files and nothing else. $(SRCS) #include some of them (gemm_data.h,
# xdma_data.h, offload_bingo_hw.h, op_test_data.h), so the -MM dependency scan must not
# run until they exist -- see the $(DEP) rule.
APP_GEN_FILES := $(PARTIAL_OUTPUTS)

PARTIAL_ELF     = $(abspath $(BUILDDIR)/$(APP).part.elf)
PARTIAL_DUMP    = $(abspath $(BUILDDIR)/$(APP).part.dump)
DEP             = $(abspath $(BUILDDIR)/$(APP).d)

ifneq (,$(filter $(INCL_DEVICE_BINARY),True true TRUE 1))
# The dir the three registration files below are written into. Assigned before the
# `| $(DEVICE_BUILDDIR)` order-only prereqs and the mkdir rule that name it: make expands a
# rule's target list at parse time, so an empty value there leaves the rule with no target
# and make silently discards it.
DEVICE_BUILDDIR           = $(abspath $(DEVICE_DIR))
PARTIAL_HOST_APP_LIST     = $(abspath $(DEVICE_DIR)/host_app_list.$(APP).tmp)
PARTIAL_HOST_APP_ORIGIN   = $(abspath $(DEVICE_DIR)/host_app_origin.$(APP).tmp)
PARTIAL_DEV_APP_LIST      = $(abspath $(DEVICE_DIR)/dev_app_list.$(APP).tmp)



PARTIAL_TMPS = $(PARTIAL_HOST_APP_LIST) $(PARTIAL_HOST_APP_ORIGIN) $(PARTIAL_DEV_APP_LIST)
PARTIAL_OUTPUTS += $(PARTIAL_ELF) $(PARTIAL_DUMP) $(PARTIAL_TMPS)
ELFS             = $(addprefix $(BUILDDIR)/$(APP)_, $(addsuffix .elf, $(DEVICE_APPS)))
DUMPS            = $(ELFS:.elf=.dump)
DWARFS           = $(ELFS:.elf=.dwarf)
BINS             = $(ELFS:.elf=.bin)
FINAL_OUTPUTS    = $(ELFS) $(DUMPS) $(DWARFS) $(BINS)
else
PARTIAL_OUTPUTS += $(PARTIAL_ELF) $(PARTIAL_DUMP)
ELF              = $(abspath $(BUILDDIR)/$(APP).elf)
DUMP             = $(abspath $(BUILDDIR)/$(APP).dump)
DWARF            = $(abspath $(BUILDDIR)/$(APP).dwarf)
BIN              = $(abspath $(BUILDDIR)/$(APP).bin)
FINAL_OUTPUTS    = $(ELF) $(DUMP) $(DWARF) $(BIN)
endif


#########
# Rules #
#########

.DEFAULT_GOAL := all
TARGET ?= all

.PHONY: all partial-build finalize-build clean FORCE
all:
ifeq ($(TARGET),all)
	$(MAKE) partial-build
	$(MAKE) finalize-build
else
	$(MAKE) $(TARGET)
endif

FORCE:

ifneq (,$(filter $(INCL_DEVICE_BINARY),True true TRUE 1))
# The three registration files below carry a FORCE prerequisite -- they must be
# re-DERIVED on every build so a renamed app or a relocated origin is never missed --
# but they are only re-WRITTEN when the derived content actually differs.
#
# Their mtime is load-bearing: device/Makefile concatenates them into host_app_origin.tmp,
# and device/apps/common.mk hangs every origin_<host>.ld off that file, so a bumped mtime
# re-generates all ~53 origin scripts, relinks all ~53 device ELFs and re-runs
# llvm-objdump + llvm-dwarfdump on each -- ~90 s of work for byte-identical output. An
# unconditional `>` bumps the mtime even when nothing changed; writing through a .new temp
# and mv-ing only on a real change keeps the re-derivation while leaving the mtime (and the
# whole downstream chain) alone.
#
# The temp is named .new, NOT .tmp: device/Makefile globs host_app_*.*.tmp, and a
# transient *.tmp would be swept into the aggregated lists.
define commit_if_changed
	@if cmp -s $@.new $@; then rm -f $@.new; else mv -f $@.new $@; fi
endef

# The three registration files are consumed POSITIONALLY: device/apps/common.mk pairs
# host_app_list.tmp line N with host_app_origin.tmp line N and dev_app_list.tmp line N.
# So all three must appear together or not at all: an app that contributes its name and
# dev-app list but no origin puts the files out of step by one line, and EVERY device ELF
# from that app onward is linked at another app's L3_ORIGIN, silently. All three therefore
# hang off $(PARTIAL_ELF) -- nothing registers unless the link produced an ELF.

# Host app list
$(PARTIAL_HOST_APP_LIST): $(PARTIAL_ELF) FORCE | $(DEVICE_BUILDDIR)
	@echo "$(APP)" > $@.new
	$(commit_if_changed)

# Host app origin (from partial ELF)
# -w: an unanchored `grep snitch_main` also matches a snitch_main_* symbol and emits a
# second line, shifting every later origin by one word. The emptiness check is explicit
# because the pipeline's status is cut's (always 0), so a missing symbol would otherwise
# be written out as a silent "0x" origin.
$(PARTIAL_HOST_APP_ORIGIN): $(PARTIAL_ELF) FORCE | $(DEVICE_BUILDDIR)
	@RELOC_ADDR=$$($(RISCV_OBJDUMP) -t $< | grep -w snitch_main | cut -c9-16); \
	if [ -z "$$RELOC_ADDR" ]; then echo "no snitch_main symbol in $<" >&2; exit 1; fi; \
	echo "0x$$RELOC_ADDR" > $@.new
	$(commit_if_changed)

# Dev app list (for each host app -> which device apps it needs)
$(PARTIAL_DEV_APP_LIST): $(PARTIAL_ELF) FORCE | $(DEVICE_BUILDDIR)
	@echo "$(DEVICE_APPS)" > $@.new
	$(commit_if_changed)
endif



# ---- Per-app failure isolation, platform guard, and build status ---------------------
#
# Platform-applicability guard (bingo_hw workloads only). A bingo workload declares the
# platform it targets (num_chiplets / num_clusters in its params.hjson) and the
# mini_compiler enforces it: guard_chiplet_count / guard_cluster_count REFUSE to emit
# $(OFFLOAD_H) when the HW actually built does not match (e.g. a 4-chiplet workload under a
# 1-chiplet cfg), print a WARNING, and exit cleanly -- "this workload does not apply to this
# platform". The build honours that: generate the header first, and if the guard declined to
# produce it, build nothing for this workload and say so. An inapplicable workload must not
# reach the compiler, or it dies on the missing include and -- since the app trees build
# their workloads as ordinary prerequisites -- takes every workload behind it down with it.
#
# `make sw` builds the whole ~96-app fleet against ONE HW cfg. Some apps legitimately do
# not fit (a 128 kiB WIDE_SPM cannot hold gemm_sweep's D_cfg tables). SW_NONFATAL=1 -- set
# by `make sw`, NOT by `single-sw`/`make apps` -- turns a failing app into an ISOLATED,
# RECORDED skip: its outputs are purged, a status shard is written, and this make exits 0 so
# the remaining apps still build. The failures show up in the build summary.
SW_NONFATAL   ?= 0
SW_STATUS_DIR ?= $(abspath $(SWDIR)/tmp/status)
APP_LOG        = $(abspath $(BUILDDIR)/$(APP).build.log)
APP_STATUS     = $(SW_STATUS_DIR)/$(APP).status

# What a FAILED or SKIPPED app must leave behind: nothing.
#
# Purging the three registration files is NOT optional. They are consumed POSITIONALLY by
# device/apps/common.mk (host_app_list line N pairs with host_app_origin line N), so an app
# that contributes a name but no origin shifts every later device ELF onto another app's
# L3_ORIGIN. "make simply didn't rewrite them" is not enough either: a STALE tmp from a
# previous successful build survives on disk and would still register an app whose ELF no
# longer exists. Deleting all three together keeps the lists in step.
PARTIAL_PURGE = $(PARTIAL_ELF) $(PARTIAL_DUMP) $(PARTIAL_TMPS)

# The platform guard (bingo workloads): guard_chiplet_count / guard_cluster_count REFUSE to
# emit $(OFFLOAD_H) when the HW actually built does not match the workload, print a warning,
# and exit ZERO -- "this workload does not apply to this platform". So a NON-ZERO exit from
# this goal is a real generator crash and must NOT be mistaken for "not applicable".
ifneq ($(strip $(OFFLOAD_H)),)
GUARD_GOAL = $(OFFLOAD_H)
GUARD_TEST = [ -f "$(OFFLOAD_H)" ]
GUARD_MSG  = platform guard did not emit $(notdir $(OFFLOAD_H)) - workload does not target this HW cfg
else
GUARD_GOAL = __app-noguard
GUARD_TEST = true
GUARD_MSG  =
endif

# $(1) = stage tag (partial|finalize)   $(2) = files to purge
define record_fail
	cat $(APP_LOG) >&2; \
	err=$$(sed -n \
	    -e "s/.*\(region .* overflowed by [0-9][0-9]* bytes\).*/\1/p" \
	    -e "s/.*\(section .* will not fit in region .*\)/\1/p" \
	    -e "s/.*\(undefined reference to .*\)/\1/p" \
	    -e "s/.*\(No rule to make target .*\)/\1/p" \
	    -e "s/.*\(error: .*\)/\1/p" $(APP_LOG) | head -1 | tr -d '\t'); \
	[ -n "$$err" ] || err="build failed (see $(APP_LOG))"; \
	rm -f $(2); \
	printf 'FAIL\t%s\t%s\t%s\n' '$(APP)' '$(1)' "$$err" > $(APP_STATUS); \
	echo "[FAIL] $(APP) ($(1)-build): $$err" >&2; \
	if [ "$(SW_NONFATAL)" != "1" ]; then exit 1; fi
endef

.PHONY: partial-build finalize-build __app-partial __app-finalize __app-noguard
__app-noguard: ;
__app-partial:  $(PARTIAL_OUTPUTS)
__app-finalize: $(FINAL_OUTPUTS)

# mkdir is folded into the SAME shell command as the $(MAKE) call, not a separate recipe
# line: make executes recipe lines containing $(MAKE) even under `-n`, but skips the others,
# so a standalone mkdir would be skipped and the `> $(APP_LOG)` redirect would then fail on a
# missing build/ directory -- breaking `make -n` for the whole tree.
partial-build:
	@mkdir -p $(BUILDDIR) $(SW_STATUS_DIR); \
	 if ! $(MAKE) --no-print-directory $(GUARD_GOAL) > $(APP_LOG) 2>&1; then \
	    $(call record_fail,partial,$(PARTIAL_PURGE)); \
	 elif ! $(GUARD_TEST); then \
	    cat $(APP_LOG); \
	    echo "[skip] $(APP): $(GUARD_MSG); nothing to build"; \
	    rm -f $(PARTIAL_PURGE); \
	    printf 'SKIP\t%s\tpartial\t%s\n' '$(APP)' '$(GUARD_MSG)' > $(APP_STATUS); \
	 elif $(MAKE) --no-print-directory __app-partial >> $(APP_LOG) 2>&1; then \
	    cat $(APP_LOG); \
	    printf 'OK\t%s\tpartial\t-\n' '$(APP)' > $(APP_STATUS); \
	 else \
	    $(call record_fail,partial,$(PARTIAL_PURGE)); \
	 fi

# No partial ELF => the app was guard-skipped or failed to link. Do NOT try to finalize.
# For an app that embeds a device binary the final ELF hangs off the DEVICE .bin, which was
# never built because the app never registered -- it would die on "No rule to make target
# ....bin" rather than on the original overflow. Leaving $(APP_STATUS) untouched preserves
# the partial verdict.
#
# A FINALIZE failure does NOT unregister: the device binaries for this app were already
# built at its origin, and pulling the tmps mid-run would desync the positional lists.
finalize-build:
	@mkdir -p $(BUILDDIR) $(SW_STATUS_DIR); \
	 if [ ! -f "$(PARTIAL_ELF)" ]; then \
	    echo "[skip] $(APP): no partial ELF (guard-skipped or link failed); skipping finalize"; \
	 elif $(MAKE) --no-print-directory __app-finalize > $(APP_LOG) 2>&1; then \
	    cat $(APP_LOG); \
	    printf 'OK\t%s\tfinalize\t-\n' '$(APP)' > $(APP_STATUS); \
	 else \
	    $(call record_fail,finalize,$(FINAL_OUTPUTS)); \
	 fi

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
	rm -rf $(PARTIAL_TMPS)

$(BUILDDIR):
	mkdir -p $@

$(DEVICE_BUILDDIR):
	mkdir -p $@

# Every target whose recipe COMPILES $(SRCS). They must all carry the header
# prerequisites the compiler discovers. ($(ELFS) and $(ELF) are set in mutually
# exclusive branches above, so exactly one of them is non-empty here.)
#
# $(DEP) names ITSELF as a target as well as the ELFs, so the .d is regenerated whenever
# any header it lists changes. That is what keeps its header set current when an #include
# is added to a HEADER rather than to $(SRCS) or $(APP_GEN_FILES).
DEP_TARGETS = $(PARTIAL_ELF) $(ELFS) $(ELF) $(DEP)

# $(APP_GEN_FILES) are prerequisites, not just $(SRCS): the -MM scan COMPILES the sources,
# which #include the app's generated headers (gemm_data.h, xdma_data.h, ...), so those must
# exist before the scan runs.
$(DEP): $(SRCS) $(APP_GEN_FILES) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -MM -MT '$(strip $(DEP_TARGETS))' $< > $@

# Read the discovered prerequisites. This is what makes an edit to a HEADER
# (host_kernel_lib.h, ara_sweep.h, ...) rebuild the ELFs that include it: $(DEP) itself only
# depends on $(SRCS) and $(APP_GEN_FILES), so the header set is known to make only through
# the .d.
#
# Include it ONLY IF IT ALREADY EXISTS. Including a file makes `make` try to REMAKE it
# during its PARSE phase -- before any normal target ordering -- and remaking $(DEP) runs
# the compiler's -MM scan over $(SRCS). Those sources #include headers that other parts of
# the build GENERATE: the app's own (gemm_data.h, offload_bingo_hw.h) and, crucially, the
# GLOBAL platform headers (host.h -> occamy.h, allocators.h -> occamy_memory_map.h) which
# `make sw` produces via its own prerequisites. On a clean tree none of them exist yet, so
# an unconditional include would run that scan before anything could generate them, and it
# would die on occamy.h.
#
# With the wildcard, a from-scratch build has no .d, nothing is included, and normal target
# ordering generates the headers and then compiles (the ELF depends on $(DEP), which is
# built at the right time). Once the app HAS been built its .d exists and every later build
# reads it. It also keeps a workload the platform guard skips -- which never compiles and so
# never gets a .d -- out of a parse-time scan of a header that will never be generated.
ifneq ($(wildcard $(DEP)),)
-include $(DEP)
endif

# Partially linked object
$(PARTIAL_ELF): $(DEP) $(LD_SRCS) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) $(SRCS) $(RISCV_LDFLAGS)  -o $@

$(PARTIAL_DUMP): $(PARTIAL_ELF) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

ifneq (,$(filter $(INCL_DEVICE_BINARY),True true TRUE 1))
# only the incldevicebinary option is enabled, we build the final host app with device binaries

define elf_rule_template =
    $$(BUILDDIR)/$$(APP)_$(1).elf: $$(DEVICE_DIR)/apps/snax/$(1)/build/$$(APP)_$(1).bin $$(DEP) $$(LD_SRCS) | $$(BUILDDIR)
		$$(RISCV_CC) $$(RISCV_CFLAGS) $$(SRCS) -DDEVICEBIN=\"$$<\" $$(RISCV_LDFLAGS)  -o $$@
endef
$(foreach f,$(DEVICE_APPS),$(eval $(call elf_rule_template,$(f))))

$(BUILDDIR)/$(APP)_%.dump: $(BUILDDIR)/$(APP)_%.elf | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

$(BUILDDIR)/$(APP)_%.dwarf: $(BUILDDIR)/$(APP)_%.elf | $(BUILDDIR)
	$(RISCV_READELF) --debug-dump $< > $@

$(BUILDDIR)/$(APP)_%.bin: $(BUILDDIR)/$(APP)_%.elf | $(BUILDDIR)
	$(RISCV_OBJCOPY) -O binary $< $@
else
# else we only build the host app without device binaries
$(ELF): $(DEP) $(LD_SRCS) | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) $(SRCS) $(RISCV_LDFLAGS) -o $@
$(DUMP): $(ELF) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@
$(DWARF): $(ELF) | $(BUILDDIR)
	$(RISCV_READELF) --debug-dump $< > $@
$(BIN): $(ELF) | $(BUILDDIR)
	$(RISCV_OBJCOPY) -O binary $< $@
endif