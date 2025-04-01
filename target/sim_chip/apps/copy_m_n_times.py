#!/usr/bin/env python3

import os
import shutil
import argparse

def copy_directory_m_n_times(input_dir, m, n):
    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} does not exist or is not a directory.")
        return
    
    os.makedirs("../bin/", exist_ok=True)
    # Loop through m and n to create copies
    for x in range(m):
        for y in range(n):
            # Define the new directory name
            output_dir = f"../bin/app_chip_{x}_{y}"
            
            # Copy the directory to the new location
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            shutil.copytree(input_dir, output_dir)
            print(f"Copied {input_dir} to {output_dir}")

def main():
    # Define argument parser
    parser = argparse.ArgumentParser(description='Copy a file m*n times with specific naming convention.')
    
    # Positional arguments
    parser.add_argument('-i', '--input-dir', required=True, help='Path to the input directory to be copied')
    parser.add_argument('-m', required=True, type=int, help='Number of rows (copies along x-axis)')
    parser.add_argument('-n', required=True, type=int, help='Number of columns (copies along y-axis)')
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    copy_directory_m_n_times(args.input_dir, args.m, args.n)

if __name__ == '__main__':
    main()
