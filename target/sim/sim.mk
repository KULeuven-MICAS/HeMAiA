
# The original common.mk file is pasted here

LOGS_DIR ?= bin/logs
UTIL_DIR ?= $(SNITCH_ROOT)/util

# SEPP packages
QUESTA_SEPP    ?=
VCS_SEPP       ?=
VERILATOR_SEPP ?=

# External executables
BENDER       ?= bender
DASM         ?= spike-dasm
VLT          ?= $(VERILATOR_SEPP) verilator
VCS          ?= $(VCS_SEPP) vcs
VERIBLE_FMT  ?= verible-verilog-format
CLANG_FORMAT ?= clang-format
VSIM         ?= $(QUESTA_SEPP) vsim
VOPT         ?= $(QUESTA_SEPP) vopt
VLOG         ?= $(QUESTA_SEPP) vlog
VLIB         ?= $(QUESTA_SEPP) vlib

# Internal executables
GENTRACE_PY      ?= $(UTIL_DIR)/trace/gen_trace.py
ANNOTATE_PY      ?= $(UTIL_DIR)/trace/annotate.py
EVENTS_PY        ?= $(UTIL_DIR)/trace/events.py
PERF_CSV_PY      ?= $(UTIL_DIR)/trace/perf_csv.py
LAYOUT_EVENTS_PY ?= $(UTIL_DIR)/trace/layout_events.py
EVENTVIS_PY      ?= $(UTIL_DIR)/trace/eventvis.py

VERILATOR_ROOT ?= $(dir $(shell $(VERILATOR_SEPP) which verilator))..
VLT_ROOT       ?= ${VERILATOR_ROOT}
VERILATOR_VERSION ?= $(shell $(VLT) --version | grep -oP 'Verilator \K\d+')

MATCH_END := '/+incdir+/ s/$$/\/*\/*/'
MATCH_BGN := 's/+incdir+//g'
SED_SRCS  := sed -e ${MATCH_END} -e ${MATCH_BGN}

VSIM_BENDER   += -t test -t rtl -t vsim
VSIM_SOURCES   = $(shell ${BENDER} script flist ${VSIM_BENDER} | ${SED_SRCS})
VSIM_BUILDDIR ?= work-vsim
VOPT_FLAGS     = +acc
VOPT_FLAGS    += +notimingchecks

# VCS_BUILDDIR should to be the same as the `DEFAULT : ./work-vcs`
# in target/snitch_cluster/synopsys_sim.setup
VCS_BENDER   += -t test -t rtl -t vcs
VCS_SOURCES   = $(shell ${BENDER} script flist ${VCS_BENDER} | ${SED_SRCS})
VCS_BUILDDIR := work-vcs

# For synthesis with DC compiler
SYN_FLIST ?= syn_flist.tcl
SYN_BENDER += -t test -t synthesis
ifeq ($(MEM_TYPE), exclude_tcsram)
	VSIM_BENDER += -t tech_cells_generic_exclude_tc_sram
	SYN_BENDER  += -t tech_cells_generic_exclude_tc_sram
	VCS_BENDER  += -t tech_cells_generic_exclude_tc_sram
endif
ifeq ($(MEM_TYPE), prep_syn_mem)
        VSIM_BENDER += -t tech_cells_generic_exclude_tc_sram
		VCS_BENDER  += -t tech_cells_generic_exclude_tc_sram
        SYN_BENDER  += -t tech_cells_generic_exclude_tc_sram
        SYN_BENDER  += -t prep_syn_mem
endif
ifeq ($(SIM_TYPE), gate_level_sim)
        VSIM_BENDER += -t gate_level_sim
		VCS_BENDER  += -t gate_level_sim
endif
SYN_SOURCES = $(shell ${BENDER} script synopsys ${SYN_BENDER})
SYN_BUILDDIR := work-syn


VLT_BENDER   += -t rtl
VLT_SOURCES   = $(shell ${BENDER} script flist ${VLT_BENDER} | ${SED_SRCS})
VLT_BUILDDIR := work-vlt

VLT_FLAGS    += --timing
VLT_FLAGS    += --x-assign fast
VLT_FLAGS    += --x-initial fast
VLT_FLAGS    += --unroll-count 256
VLT_FLAGS    += -Wno-BLKANDNBLK
VLT_FLAGS    += -Wno-LITENDIAN
VLT_FLAGS    += -Wno-CASEINCOMPLETE
VLT_FLAGS    += -Wno-CMPCONST
VLT_FLAGS    += -Wno-WIDTH
VLT_FLAGS    += -Wno-WIDTHCONCAT
VLT_FLAGS    += -Wno-UNSIGNED
VLT_FLAGS    += -Wno-UNOPTFLAT
VLT_FLAGS    += -Wno-fatal
VLT_FLAGS    += -Wno-SYMRSVDWORD
VLT_FLAGS    += -Wno-BLKLOOPINIT

VLT_FLAGS    += +define+SYNTHESIS


VLT_CXXSTD_FLAGS += -std=c++20 -O3 -march=native -flto -pthread -latomic

VLT_CFLAGS   += ${VLT_CXXSTD_FLAGS} -I ${VLT_BUILDDIR} -I $(VLT_ROOT)/include -I $(VLT_ROOT)/include/vltstd -I $(VLT_FESVR)/include -I ${MKFILE_DIR}/test

ANNOTATE_FLAGS ?= -q --keep-time

# We need a recent LLVM installation (>11) to compile Verilator.
# We also need to link the binaries with LLVM's libc++.
# Define CLANG_PATH to be the path of your Clang installation.

ifneq (${CLANG_PATH},)
    CLANG_CC       := $(CLANG_PATH)/bin/clang
    CLANG_CXX      := $(CLANG_PATH)/bin/clang++
    CLANG_CXXFLAGS := -nostdinc++ -isystem $(CLANG_PATH)/include/c++/v1
    CLANG_LDFLAGS  := -nostdlib++ -fuse-ld=lld -L ${CLANG_PATH}/lib -Wl,-rpath,${CLANG_PATH}/lib -lc++
else
    CLANG_CC       ?= clang
    CLANG_CXX      ?= clang++
    CLANG_CXXFLAGS := ""
    CLANG_LDFLAGS  := ""
endif

# If requested, build verilator with LLVM and add llvm c/ld flags
ifeq ($(VLT_USE_LLVM),ON)
    CC         = $(CLANG_CC)
    CXX        = $(CLANG_CXX)
    CFLAGS     = $(CLANG_CXXFLAGS)
    CXXFLAGS   = $(CLANG_CXXFLAGS)
    LDFLAGS    = $(CLANG_LDFLAGS)
    VLT_FLAGS += --compiler clang
    VLT_FLAGS += -CFLAGS "${CLANG_CXXFLAGS}"
    VLT_FLAGS += -LDFLAGS "${CLANG_LDFLAGS}"
endif

VLOGAN_FLAGS := -assert svaext
VLOGAN_FLAGS += -assert disable_cover
VLOGAN_FLAGS += -kdb
VLOGAN_FLAGS += -timescale=1ns/1ps
VLOGAN_FLAGS += -override_timescale=1ns/1ps
# VLOGAN_FLAGS += -work ./work-vcs
VHDLAN_FLAGS += -kdb
VHDLAN_FLAGS += -timescale=1ns/1ps
VHDLAN_FLAGS += -override_timescale=1ns/1ps
# VHDLAN_FLAGS += -work ./work-vcs

#############
# Verilator #
#############
# Takes the top module name as an argument.
define VERILATE
	mkdir -p $(dir $@)
	$(BENDER) script verilator ${VLT_BENDER} > $(dir $@)files
	+$(VLT) \
		--Mdir $(dir $@) -f $(dir $@)files $(VLT_FLAGS) \
		-j $(shell nproc) --cc --build --top-module $(1)
	touch $@
endef


##########
# Traces #
##########

DASM_TRACES      = $(shell (ls $(LOGS_DIR)/trace_chip_??_hart_*.dasm 2>/dev/null))
TXT_TRACES       = $(shell (echo $(DASM_TRACES) | sed 's/\.dasm/\.txt/g'))
PERF_TRACES      = $(shell (echo $(DASM_TRACES) | sed 's/.dasm/_perf.json/g'))
ANNOTATED_TRACES = $(shell (echo $(DASM_TRACES) | sed 's/\.dasm/\.s/g'))
DIFF_TRACES      = $(shell (echo $(DASM_TRACES) | sed 's/\.dasm/\.diff/g'))

GENTRACE_OUTPUTS = $(TXT_TRACES) $(PERF_TRACES)
ANNOTATE_OUTPUTS = $(ANNOTATED_TRACES)
PERF_CSV         = $(LOGS_DIR)/perf.csv
EVENT_CSV        = $(LOGS_DIR)/event.csv
TRACE_CSV        = $(LOGS_DIR)/trace.csv
TRACE_JSON       = $(LOGS_DIR)/trace.json

.PHONY: traces annotate perf-csv event-csv layout
traces: $(GENTRACE_OUTPUTS)
annotate: $(ANNOTATE_OUTPUTS)
diff-trace: $(DIFF_TRACES)
perf-csv: $(PERF_CSV)
event-csv: $(EVENT_CSV)
layout: $(TRACE_CSV) $(TRACE_JSON)

$(LOGS_DIR)/%.txt $(LOGS_DIR)/%_perf.json: $(LOGS_DIR)/%.dasm $(GENTRACE_PY)
	@CHIP=$(word 3,$(subst _, ,$*)) && \
	HART=$(word 5,$(subst _, ,$*)) && \
	echo "Processing Chip $$CHIP Hart $$HART" && \
	$(DASM) < $< | $(PYTHON) $(GENTRACE_PY) --permissive -d $(LOGS_DIR)/trace_chip_$$CHIP\_hart_$$HART\_perf.json > $(LOGS_DIR)/trace_chip_$$CHIP\_hart_$$HART.txt

# Generate source-code interleaved traces for all harts.
BINARY ?= $(shell cat apps/.rtlbinary)

$(LOGS_DIR)/%.s: $(LOGS_DIR)/%.txt $(ANNOTATE_PY)
	$(PYTHON) $(ANNOTATE_PY) $(ANNOTATE_FLAGS) -o $@ $(BINARY) $<
$(LOGS_DIR)/%.diff: $(LOGS_DIR)/%.txt $(ANNOTATE_PY)
	$(PYTHON) $(ANNOTATE_PY) $(ANNOTATE_FLAGS) -o $@ $(BINARY) $< -d

$(PERF_CSV): $(PERF_TRACES) $(PERF_CSV_PY)
	$(PYTHON) $(PERF_CSV_PY) -o $@ -i $(PERF_TRACES)

$(EVENT_CSV): $(PERF_TRACES) $(PERF_CSV_PY)
	$(PYTHON) $(PERF_CSV_PY) -o $@ -i $(PERF_TRACES) --filter tstart tend

$(TRACE_CSV): $(EVENT_CSV) $(LAYOUT_FILE) $(LAYOUT_EVENTS_PY)
	$(PYTHON) $(LAYOUT_EVENTS_PY) $(LAYOUT_EVENTS_FLAGS) $(EVENT_CSV) $(LAYOUT_FILE) -o $@

$(TRACE_JSON): $(TRACE_CSV) $(EVENTVIS_PY)
	$(PYTHON) $(EVENTVIS_PY) -o $@ $(TRACE_CSV)
