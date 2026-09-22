# Fanchen Kong <fanchen.kong@kuleuven.be>
# Helper functions for address transformation mirroring chip_id.h

def _check_xdma_size_aligned(size: int, kernel_name: str, xdma_width: int = 64) -> None:
    """Validate that an xDMA 1D transfer size is a multiple of the datapath width.

    xdma_width must match XDMA_WIDTH in device/runtime/snax/xdma/snax_xdma_addr.h
    (default 64). xdma_memcpy_1d_full_addr returns -1 (without programming any
    descriptor) for non-aligned sizes; xdma_start() then kicks off the engine
    with stale state and the wait loop hangs forever — so catch it here, at
    IR-build time, with a clear message.
    """
    if not isinstance(size, int):
        raise TypeError(f"{kernel_name}: size must be int, got {type(size).__name__}={size!r}")
    if size <= 0:
        raise ValueError(f"{kernel_name}: size must be positive, got {size}")
    if size % xdma_width != 0:
        raise ValueError(
            f"{kernel_name}: size={size} bytes is not a multiple of XDMA_WIDTH="
            f"{xdma_width}. xdma_memcpy_1d_full_addr would silently fail and "
            f"xdma_start() would hang waiting on the completion CSR. Pad the "
            f"transfer or adjust the workload tiling so size % {xdma_width} == 0."
        )


def chip_loc_to_chip_id(x: int, y: int) -> int:
    """
    Convert chip coordinates (x, y) to a chip ID.
    Equivalent to C: (x << 4) | (y & 0x0F)
    """
    return (x << 4) | (y & 0x0F)

def get_chip_baseaddress(chip_id: int) -> int:
    """
    Get the base address for a given chip ID.
    Equivalent to C: ((uint64_t)chip_id << 40)
    """
    return chip_id << 40

def chiplet_addr_transform_full(chip_id: int, addr: int) -> int:
    """
    Transform a local address to a global address for a specific chip ID.
    Equivalent to C: get_chip_baseaddress(chip_id) | addr
    """
    return get_chip_baseaddress(chip_id) | addr

def chiplet_addr_transform_loc(chip_loc_x: int, chip_loc_y: int, addr: int) -> int:
    """
    Transform a local address to a global address for specific chip coordinates.
    Equivalent to C: get_chip_baseaddress(chip_id) | addr
    """
    chip_id = chip_loc_to_chip_id(chip_loc_x, chip_loc_y)
    return get_chip_baseaddress(chip_id) | addr
