#!/usr/bin/env python3
import os

def bin2preload(input_file, output_dir):
    # Constants
    NUM_BANKS = 32
    BANK_WIDTH = 64  # 64 bits = 8 bytes
    TOTAL_WIDTH = NUM_BANKS * BANK_WIDTH  # 2048 bits = 256 bytes

    # Load the .bin file
    with open(input_file, 'rb') as f:
        binary_data = f.read()

    # Initialize 16 memory banks
    memory_banks = [[] for _ in range(NUM_BANKS)]

    # Process the binary data
    for i in range(0, len(binary_data), TOTAL_WIDTH // 8):  # 512 bits = 64 bytes
        chunk = binary_data[i:i + TOTAL_WIDTH // 8]  # Read 64 bytes (512 bits)
        if len(chunk) < TOTAL_WIDTH // 8:
            chunk += b'\x00' * (TOTAL_WIDTH // 8 - len(chunk))  # Pad with zeros if necessary

        # Split the chunk into 16 × 32-bit words and distribute to banks
        for bank_id in range(NUM_BANKS):
            start = bank_id * (BANK_WIDTH // 8)  # 32 bits = 4 bytes
            end = start + (BANK_WIDTH // 8)
            word = chunk[start:end]  # Extract 32-bit word
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
        print(f"Generated {output_file} with {len(memory_banks[bank_id])} words.")

    print("All banks generated successfully.")
