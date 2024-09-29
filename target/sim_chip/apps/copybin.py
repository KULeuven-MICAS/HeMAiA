#!/usr/bin/env python3

import os
import shutil
import argparse

def copy_file_m_n_times(input_file, m, n):
    # Check if the input file exists
    if not os.path.isfile(input_file):
        print(f"Error: {input_file} does not exist.")
        return
    
    os.makedirs("../bin/", exist_ok=True)
    # Loop through m and n to create copies
    for x in range(m):
        for y in range(n):
            # Define the new file name
            output_file = f"../bin/app_chip_{x}_{y}.bin"
            
            # Copy the file to the new location with the new name
            shutil.copy(input_file, output_file)
            print(f"Copied {input_file} to {output_file}")

def main():
    # Define argument parser
    parser = argparse.ArgumentParser(description='Copy a file m*n times with specific naming convention.')
    
    # Positional arguments
    parser.add_argument('-i', '--input-file', required=True, help='Path to the input file to be copied')
    parser.add_argument('-m', required=True, type=int, help='Number of rows (copies along x-axis)')
    parser.add_argument('-n', required=True, type=int, help='Number of columns (copies along y-axis)')
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    copy_file_m_n_times(args.input_file, args.m, args.n)

if __name__ == '__main__':
    main()
