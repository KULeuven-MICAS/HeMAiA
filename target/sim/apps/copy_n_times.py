#!/usr/bin/env python3

import os
import shutil
import argparse
import hjson
from jsonref import JsonRef
def read_json_file(file):
    try:
        srcfull = file.read()
        obj = hjson.loads(srcfull, use_decimal=True)
        obj = JsonRef.replace_refs(obj)
    except ValueError:
        raise SystemExit(sys.exc_info()[1])
    return obj

def copy_directory_n_times(input_dir, output_dir, folder_names):
    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} does not exist or is not a directory.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    # Loop through m and n to create copies
    for name in folder_names:
        # Define the new directory name
        app_dir = f"{output_dir}/{name}"
        
        # Copy the directory to the new location
        if os.path.exists(app_dir):
            shutil.rmtree(app_dir)
        shutil.copytree(input_dir, app_dir)
        print(f"Copied {input_dir} to {app_dir}")

def main():
    # Define argument parser
    parser = argparse.ArgumentParser(description='Copy a file m*n times with specific naming convention.')
    
    # Positional arguments
    parser.add_argument('-i', '--input-dir', required=True, help='Path to the input directory to be copied')
    parser.add_argument('-o', '--output-dir', required=True, help='Path to the output directory to be copied')
    parser.add_argument("--cfg",
                        "-c",
                        metavar="file",
                        type=argparse.FileType('r'),
                        required=True,
                        help="A cluster configuration file")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Read HJSON description of System.
    with args.cfg as file:
        occamy_cfg = read_json_file(file)
    folder_names = []
    for chip in occamy_cfg['hemaia_multichip']['testbench_cfg']['hemaia_compute_chip']:
        folder_names.append("app_chip_" + str(chip["coordinate"][0]) + "_" + str(chip["coordinate"][1]))
    # Call the function with parsed arguments
    copy_directory_n_times(args.input_dir, args.output_dir, folder_names)

if __name__ == '__main__':
    main()
