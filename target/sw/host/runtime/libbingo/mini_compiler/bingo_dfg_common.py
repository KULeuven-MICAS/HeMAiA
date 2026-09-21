# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Names shared by bingo_dfg and its mixins.
#
# These live here rather than in bingo_dfg so that a mixin never has to import bingo_dfg:
# bingo_dfg imports the mixins to build the class, so the reverse import would be a cycle.

import subprocess
import sys


def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    import site
    from importlib import reload
    reload(site)


# Which accelerator a BINGO device kernel drives, keyed on the token right after
# "_kernel_". The name is the only thing the DFG and the device-side SNAX_EXPORT_FUNC
# registry agree on, so it is what the placement is checked against.
#
# A kernel that touches no accelerator maps to None and is not placed: dummy, exit,
# sync_probe, check_results, and the scalar helpers (pack_fa_partial) which run wherever
# their consumer is. "idma" maps to the "dm" role because that is the role name for the
# hart carrying the DMA ISA -- a dm* instruction on any other hart traps.
_ENGINE_BY_KERNEL_TOKEN = {
    "gemm": "gemm",
    "simd": "simd",
    "xdma": "xdma",
    "idma": "dm",
}


def _engine_of_kernel(kernel_name):
    """The engine a __snax_bingo_kernel_* name drives, or None if it drives none."""
    if not kernel_name.startswith("__snax_bingo_kernel_"):
        return None            # the older __snax_kernel_* family is not BINGO-dispatched
    head = kernel_name[len("__snax_bingo_kernel_"):].split("_", 1)[0]
    return _ENGINE_BY_KERNEL_TOKEN.get(head)


# The task list the host hands the scheduler stays a uint64_t array however wide the
# descriptor gets, so a descriptor always occupies a whole number of these words.
BINGO_TASK_LIST_WORD_BITS = 64
BINGO_TASK_LIST_WORD_MASK = (1 << BINGO_TASK_LIST_WORD_BITS) - 1
