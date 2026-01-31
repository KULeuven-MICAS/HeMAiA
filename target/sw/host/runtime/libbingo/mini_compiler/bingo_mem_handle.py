# Fanchen Kong <fanchen.kong@kuleuven.be>
# This file defines three types of memory abstractions used in Bingo DFG compilation:
# 1. BingoMemAlloc:
#    - Represents dynamic memory allocation on the heap (e.g., via malloc-like calls).
#    - The compiler will generate allocation calls (bingo_l1_alloc, bingo_l3_alloc) for these handles.
#    - Addresses are determined at runtime.
#
# 2. BingoMemSymbol:
#    - Represents existing C variables or symbols in the host/device code (e.g., `uint32_t A[16]`).
#    - Used when the data is already allocated statically or manages externally.
#    - Supports an optional `offset` representing the ABSOLUTE BYTE OFFSET from the symbol's base address.
#    - Unlike array indexing in C which scales by type size, this offset is always in bytes.
#      Examples:
#      - For `uint8_t A[]`: `BingoMemSymbol("A", offset=8)` accesses address `(uintptr_t)A + 8`.
#        This corresponds to `&A[8]`.
#      - For `uint32_t B[]`: `BingoMemSymbol("B", offset=8)` accesses address `(uintptr_t)B + 8`.
#        This corresponds to `&B[2]` (since 2 elements * 4 bytes/element = 8 bytes).
#    - The compiler emits references to these symbol names directly with address arithmetic.
#
# 3. BingoMemFixedAddr:
#    - Represents a hardcoded, absolute physical address.
#    - Used for memory-mapped peripherals or fixed memory regions (like MemPool).
#    - The compiler emits the raw integer literal (e.g., `0x80001000`).

class BingoMemAlloc:
    """
    Represents a memory allocation request (handle) that will be resolved to a runtime address in C.
    """
    def __init__(self, name: str, size: int, mem_level: str = "L3", chip_id: int = 0, cluster_id: int = 0):
        """
        :param name: Logical name for the buffer (used for variable naming in generated C).
        :param size: Size in bytes.
        :param mem_level: "L1", "L2", or "L3".
        :param chip_id: Required for L1/L2.
        :param cluster_id: Required for L1.
        """
        self.name = name
        self.size = size
        self.mem_level = mem_level
        self.chip_id = chip_id
        self.cluster_id = cluster_id
    
    def get_c_var_name(self):
        return f"ptr_{self.name}"

    def __repr__(self):
        return f"BingoMemAlloc(name='{self.name}', size={self.size}, level='{self.mem_level}')"

class BingoMemSymbol:
    """
    Represents an existing C variable (symbol) that provides the address.
    """
    def __init__(self, symbol_name: str, offset: int = 0):
        """
        :param symbol_name: Name of the C variable/symbol.
        :param offset: Byte offset to add to the symbol address.
        """
        self.symbol_name = symbol_name
        self.offset = offset

    def __repr__(self):
        parts = [f"name='{self.symbol_name}'"]
        if self.offset != 0:
            parts.append(f"offset={self.offset}")
        return f"BingoMemSymbol({', '.join(parts)})"

class BingoMemFixedAddr:
    """
    Represents an absolute memory address provided as a hex/int value.
    """
    def __init__(self, address: int):
        self.address = address

    def __repr__(self):
        return f"BingoMemFixedAddr(addr=0x{self.address:x})"
