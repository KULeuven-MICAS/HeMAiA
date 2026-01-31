# Fanchen Kong <fanchen.kong@kuleuven.be>
# Helper functions for address transformation mirroring chip_id.h

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
