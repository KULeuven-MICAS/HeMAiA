include ../toolchain.mk
TARGET ?= all
# Get the lib name under the /snax
LIBDIR = $(shell find . -mindepth 1 -maxdepth 1 -type d ! -name build)
MKFILE_DIR  := $(shell pwd)

OBJS = $(wildcard ./*/build/*.o)
SNAX_LIB = $(BUILDDIR)/libsnaxRuntime.a
SNAX_LIB_DUMP = $(BUILDDIR)/libsnaxRuntime.dump

BUILDDIR    = $(abspath build/)

$(BUILDDIR):
	mkdir -p $@

.PHONY: all clean $(LIBDIR) 

all: $(LIBDIR) $(SNAX_LIB) $(SNAX_LIB_DUMP)

$(LIBDIR): 
	$(MAKE) -C $@ $(TARGET)

$(SNAX_LIB): $(OBJS) | $(BUILDDIR)
	$(RISCV_AR) $(RISCV_ARFLAGS) $@ $^

$(SNAX_LIB_DUMP): $(SNAX_LIB) | $(BUILDDIR)
	$(RISCV_OBJDUMP) -D $< > $@

clean: 
	rm -rf $(BUILDDIR)
	rm -rf $(wildcard ./*/build)
	