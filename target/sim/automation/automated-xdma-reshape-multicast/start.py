import os
import subprocess
import shutil
import hjson
from concurrent.futures import ThreadPoolExecutor
import argparse
import re

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
hemaia_root_path = os.path.abspath(os.path.join(script_dir, "../../../.."))
param_path = os.path.abspath(os.path.join(script_dir, "../../../sw/device/apps/snax/snax-xdma-reshape-multicast/data/params.hjson"))
binary_path_1 = os.path.abspath(os.path.join(script_dir, "../../../sim/bin/occamy_chip.vsim"))
binary_path_2 = os.path.abspath(os.path.join(script_dir, "../../../sim/work-vsim"))
app_path = os.path.abspath(os.path.join(script_dir, "../../../sim/bin/app_chip_0_0"))

num_threads = 16

cfg_list = []
cfg_list.append({
    "M": 2048,
    "N": 192,
    "input_layout": "MNM16N8",
    "output_layout": "MNM8N8",
    "multicast_num": 8
})

cfg_list.append({
    "M": 2048,
    "N": 128,
    "input_layout": "MNM16N8",
    "output_layout": "MNM8N8",
    "multicast_num": 8
})

cfg_list.append({
    "M": 2048,
    "N": 512,
    "input_layout": "MNM16N8",
    "output_layout": "MNM16N8",
    "multicast_num": 8
})

cfg_list.append({
    "M": 4096,
    "N": 192,
    "input_layout": "MNM16N8",
    "output_layout": "MNM64N16",
    "multicast_num": 1
})

cfg_list.append({
    "M": 4096,
    "N": 128,
    "input_layout": "MNM16N8",
    "output_layout": "MNM64N16",
    "multicast_num": 1
})

cfg_list.append({
    "M": 4096,
    "N": 512,
    "input_layout": "MNM16N8",
    "output_layout": "MNM16N8",
    "multicast_num": 8
})

parser = argparse.ArgumentParser()
parser.add_argument(
    "--phase",
    type=int,
    default=1,
    help="Phase number to execute (default: 1)"
)
args = parser.parse_args()
phase = args.phase

# Load the existing params from params.hjson (so we only modify the needed keys)
with open(param_path, 'r') as f:
    param_data = hjson.load(f)


# Create or overwrite the result.csv file
result_csv_path = "result.csv"
with open(result_csv_path, 'w') as result_csv:
    # Write the header row
    result_csv.write("M,N,input_layout,output_layout,multicast_num,idma,xdma1,xdma2\n")

# The function needs to be parallelized
def process_configuration(folder_name,cfg):
    subprocess.run("./bin/occamy_chip.vsim", shell=True, cwd=folder_name, check=True)
    with open(os.path.join(folder_name, "bin/uart_0_0.log"), "r") as log_file:
        # Get the IDMA Experiment results
        log_file.seek(0)
        M = cfg["M"]
        N = cfg["N"]
        input_layout = cfg["input_layout"]
        output_layout = cfg["output_layout"]
        multicast_num = cfg["multicast_num"]
        idma_cycles = "N/A"
        xdma1_cycles = "N/A"
        xdma2_cycles = "N/A"
        pattern = re.compile(r"The IDMA copy to (\d+) dest is finished in (\d+) cycles")
        for ln in log_file:
            match = pattern.search(ln)
            if match:
                idma_cycles = match.group(2)

        pattern = re.compile(r"The XDMA copy to (\d+) dest is finished in (\d+) cycles")
        for ln in log_file:
            match = pattern.search(ln)
            if match:
                xdma1_cycles = match.group(2)

        pattern = re.compile(r"The XDMA multicast to (\d+) dest is finished in (\d+) cycles")
        for ln in log_file:
            match = pattern.search(ln)
            if match:
                xdma2_cycles = match.group(2)

        with open(result_csv_path, "a") as result_file:
            result_file.write(f"{M},{N},{input_layout},{output_layout},{multicast_num},{idma_cycles},{xdma1_cycles},{xdma2_cycles}\n")
    # 5. Delete the entire folder
    # shutil.rmtree(folder_name)

# Use ThreadPoolExecutor to run configurations concurrently
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for cfg in cfg_list:
        # 1. Prepare folder name based on the configuration
        folder_name = f"M_{cfg['M']}_N_{cfg['N']}_L_{cfg['input_layout']}_{cfg['output_layout']}_B_{cfg['multicast_num']}"
        if phase == 1:
            os.makedirs(folder_name, exist_ok=True)

            # 2. Modify the params.hjson values in place
            param_data["M"] = cfg["M"]
            param_data["N"] = cfg["N"]
            param_data["input_layout"] = cfg["input_layout"]
            param_data["output_layout"] = cfg["output_layout"]
            param_data["multicast_num"] = cfg["multicast_num"]

            with open(param_path, 'w') as f:
                hjson.dump(param_data, f)

            # 3a. Execute "make sw" and then "make apps" in the apps folder
            subprocess.run(["make", "sw", "CFG_OVERRIDE=target/rtl/cfg/hemaia_noc.hjson", "-j"], cwd=hemaia_root_path, check=True)
            subprocess.run(["make", "apps", "DEVICE_APP=snax-xdma-reshape-multicast"], cwd=hemaia_root_path, check=True)
            # 4. Copy the "snax-xdma-multicast" folder to the new folder
            source_dir = app_path
            target_dir = os.path.join(folder_name + "/bin", "app_chip_0_0")
            os.makedirs(os.path.dirname(target_dir), exist_ok=True)
            shutil.copytree(source_dir, target_dir)

            # 4. Copy the simulation binary into the new folder
            dest_binary_path_1 = os.path.join(folder_name + "/bin", os.path.basename(binary_path_1))
            os.makedirs(os.path.dirname(dest_binary_path_1), exist_ok=True)
            shutil.copyfile(binary_path_1, dest_binary_path_1)
            os.chmod(dest_binary_path_1, 0o755)

            dest_binary_path_2 = os.path.join(folder_name, os.path.basename(binary_path_2))
            shutil.copytree(binary_path_2, dest_binary_path_2)
        else:
            futures.append(executor.submit(process_configuration, folder_name, cfg))

    # Wait for all threads to complete
    for future in futures:
        future.result()

print("All configurations have been processed.")
