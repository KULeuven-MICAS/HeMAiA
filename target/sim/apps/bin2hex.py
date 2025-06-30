import argparse
import bin2preload
import os

# Convert a binary file to a hex string and save it to a text file
def binary_to_hex(input_binary_file, output_text_file, chunk_size):
    try:
        # Open the binary file in read-binary mode
        with open(input_binary_file, 'rb') as bin_file:
            binary_data = bin_file.read()  # Read the entire binary file

        # Initialize a list to hold the lines
        lines = []

        total_length = len(binary_data)
        index = 0

        # Process the data in chunks of 64 bytes
        while index < total_length:
            chunk = binary_data[index:index + chunk_size]

            # Reverse the data in the chunk (MSB to LSB within the line)
            reversed_chunk = chunk[::-1]

            # Convert the reversed chunk to a hexadecimal string
            hex_chunk = reversed_chunk.hex()

            # For the last chunk, if it's less than 64 bytes, pad zeros at the end
            if len(chunk) < chunk_size:
                pad_length = chunk_size - len(chunk)
                # Pad with zeros at the end to make up to 64 bytes (128 hex digits)
                hex_chunk = '00' * pad_length + hex_chunk

            # Add the hex_chunk to the list of lines
            lines.append(hex_chunk)
            index += chunk_size

        # Write the lines to the output text file
        with open(output_text_file, 'w') as text_file:
            for line in lines:
                text_file.write(line + '\n')

        print(f"Hex data has been written to {output_text_file}")

    except FileNotFoundError:
        print(f"Error: File {input_binary_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def convert_all_bin_to_hex(directory):
    # Traverse the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.bin'):
                input_file = os.path.join(root, file)
                output_dir = f"{input_file.rsplit('.', 1)[0]}"
                bin2preload.bin2preload(input_file, output_dir)

def convert_bin_to_hex(input_file):
    output_dir = f"{input_file.rsplit('.', 1)[0]}"
    bin2preload.bin2preload(input_file, output_dir)
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert program bin in strings and save it to a text file.")
    parser.add_argument(
        '-i', '--input',
        type=str,
        help="Path to the binary file."
    )
    # Parse arguments
    args = parser.parse_args()

    # Execute the task
    # convert_all_bin_to_hex(args.input)
    convert_bin_to_hex(args.input)
if __name__ == "__main__":
    main()
