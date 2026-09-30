#!/usr/bin/env python3
import os

def l3_banks(default=16):
    """spm_wide.banks of the cfg being built (target/rtl/cfg/lru.hjson, the copy the build
    writes): the testharness loads one bank_N.hex per L3 bank (load_binary.sv.tpl, mem_bank)."""
    import re
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rtl", "cfg",
                       "lru.hjson")
    try:
        text = open(cfg).read()
    except OSError:
        return default
    m = re.search(r"spm_wide\s*:\s*\{[^}]*?\bbanks\s*:\s*(\d+)", text, re.S)
    return int(m.group(1)) if m else default


def bin2preload(input_file, output_dir):
    # Constants
    # the memory chiplet's image keeps its 16 banks (load_binary.sv.tpl mem_chip_bank_count)
    NUM_BANKS = 16 if os.path.basename(input_file).startswith("mempool") else l3_banks()
    BANK_WIDTH = 64  # 64 bits = 8 bytes
    TOTAL_WIDTH = NUM_BANKS * BANK_WIDTH  # 1024 bits = 128 bytes

    # Load the .bin file
    with open(input_file, 'rb') as f:
        binary_data = f.read()

    # Initialize 16 memory banks
    memory_banks = [[] for _ in range(NUM_BANKS)]

    # Process the binary data
    for i in range(0, len(binary_data), TOTAL_WIDTH // 8):  # 1024 bits = 128 bytes
        chunk = binary_data[i:i + TOTAL_WIDTH // 8]  # Read 128 bytes (1024 bits)
        if len(chunk) < TOTAL_WIDTH // 8:
            chunk += b'\x00' * (TOTAL_WIDTH // 8 - len(chunk))  # Pad with zeros if necessary

        # Split the chunk into 16 x 64-bit words and distribute to banks
        for bank_id in range(NUM_BANKS):
            start = bank_id * (BANK_WIDTH // 8)  # 64 bits = 8 bytes
            end = start + (BANK_WIDTH // 8)
            word = chunk[start:end]  # Extract 64-bit word
            word_int = int.from_bytes(word, byteorder='little')  # Convert to integer
            memory_banks[bank_id].append(f"{word_int:08X}")  # Format as 8-digit hex string

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write each bank's data to separate .cde files
    for bank_id in range(NUM_BANKS):
        output_file = f'{output_dir}/bank_{bank_id}.hex'
        with open(output_file, 'w') as f:
            for word in memory_banks[bank_id]:
                f.write(word + '\n')  # Write each 32-bit word as a line in hex
        print(f"Generated {output_file} with {len(memory_banks[bank_id])} words.", flush=True)

    print("All banks generated successfully.", flush=True)
