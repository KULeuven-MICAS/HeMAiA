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
param_path = os.path.abspath(os.path.join(script_dir, "../../../sw/device/apps/snax/snax-xdma-multicast-exploration/data/params.hjson"))
binary_path_1 = os.path.abspath(os.path.join(script_dir, "../../../sim/bin/occamy_chip.vsim"))
binary_path_2 = os.path.abspath(os.path.join(script_dir, "../../../sim/work-vsim"))
app_path = os.path.abspath(os.path.join(script_dir, "../../../sim/bin/app_chip_0_0"))

num_threads = 16
num_dests = [2, 4, 6, 8, 10, 12]
num_repetitions = 32

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
    result_csv.write("type,num_dest,cycles,hops\n")

# The function needs to be parallelized
def process_configuration(folder_name):
    subprocess.run("./bin/occamy_chip.vsim", shell=True, cwd=folder_name, check=True)
    with open(os.path.join(folder_name, "bin/uart_0_0.log"), "r") as log_file:
        # Get the IDMA Experiment results
        log_file.seek(0)
        typ = "idma"
        dest = "N/A"
        cycles = "N/A"
        hops = "N/A"
        pattern = re.compile(r"The IDMA copy to (\d+) dest is finished in (\d+) cycles with (\d+) hops")
        for ln in log_file:
            match = pattern.search(ln)
            if match:
                dest = match.group(1)
                cycles = match.group(2)
                hops = match.group(3)
        with open(result_csv_path, "a") as result_file:
            result_file.write(f"{typ},{dest},{cycles},{hops}\n")

        # Get the XDMA Experiment results
        log_file.seek(0)
        typ = "xdma_normal"
        dest = "N/A"
        cycles = "N/A"
        hops = "N/A"
        pattern = re.compile(r"The XDMA normal copy to (\d+) dest is finished in (\d+) cycles with (\d+) hops")
        for ln in log_file:
            match = pattern.search(ln)
            if match:
                dest = match.group(1)
                cycles = match.group(2)
                hops = match.group(3)
        with open(result_csv_path, "a") as result_file:
            result_file.write(f"{typ},{dest},{cycles},{hops}\n")

        # Get the XDMA Experiment results
        log_file.seek(0)
        typ = "xdma_optimal"
        dest = "N/A"
        cycles = "N/A"
        hops = "N/A"
        pattern = re.compile(r"The XDMA optimal copy to (\d+) dest is finished in (\d+) cycles with (\d+) hops")
        for ln in log_file:
            match = pattern.search(ln)
            if match:
                dest = match.group(1)
                cycles = match.group(2)
                hops = match.group(3)
        with open(result_csv_path, "a") as result_file:
            result_file.write(f"{typ},{dest},{cycles},{hops}\n")

        # Get the Ideal Experiment results
        log_file.seek(0)
        typ = "ideal"
        dest = "N/A"
        cycles = "N/A"
        hops = "N/A"
        pattern = re.compile(r"The ideal broadcast to (\d+) dest has (\d+) hops")
        for ln in log_file:
            match = pattern.search(ln)
            if match:
                dest = match.group(1)
                hops = match.group(3)
        with open(result_csv_path, "a") as result_file:
            result_file.write(f"{typ},{dest},{cycles},{hops}\n")
    # 5. Delete the entire folder
    shutil.rmtree(folder_name)

# Use ThreadPoolExecutor to run configurations concurrently
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for i in range(num_repetitions):
        for num_dest in num_dests:
            # 1. Prepare folder name based on the configuration
            folder_name = f"S_{num_dest}_R_{i}"
            if phase == 1:
                os.makedirs(folder_name, exist_ok=True)

                # 2. Modify the params.hjson values in place
                param_data["multicast_num"] = num_dest
                with open(param_path, 'w') as f:
                    hjson.dump(param_data, f)

                # 3a. Execute "make sw" and then "make apps" in the apps folder
                subprocess.run(["make", "sw", "CFG_OVERRIDE=target/rtl/cfg/hemaia_noc.hjson", "-j"], cwd=hemaia_root_path, check=True)
                subprocess.run(["make", "apps", "DEVICE_APP=snax-xdma-multicast-exploration"], cwd=hemaia_root_path, check=True)
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
                futures.append(executor.submit(process_configuration, folder_name))

    # Wait for all threads to complete
    for future in futures:
        future.result()

print("All configurations have been processed.")
