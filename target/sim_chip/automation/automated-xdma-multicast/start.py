import os
import subprocess
import shutil
import hjson
from concurrent.futures import ThreadPoolExecutor

param_path = "/users/micas/ydeng/Desktop/SNAX/HeMAiA/target/sim/sw/device/apps/snax/snax-xdma-multicast/data/params.hjson"
binary_path = "/users/micas/ydeng/Desktop/SNAX/HeMAiA/target/sim_chip/bin/occamy_chip.vlt"
hemaia_root_path = "/users/micas/ydeng/Desktop/SNAX/HeMAiA"

num_threads = 24
sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 786432]

# Load the existing params from params.hjson (so we only modify the needed keys)
with open(param_path, 'r') as f:
    param_data = hjson.load(f)


# Create or overwrite the result.csv file
result_csv_path = "result.csv"
with open(result_csv_path, 'w') as result_csv:
    # Write the header row
    result_csv.write("size,idma_copy,xdma_copy,idma_broadcast_1,xdma_broadcast_1,idma_broadcast_2,xdma_broadcast_2,idma_broadcast_3,xdma_broadcast_3,idma_broadcast_4,xdma_broadcast_4\n")

# The function needs to be parallelized
def process_configuration(folder_name, size):
    # 3a. Copy hardware binary into the new folder
    dest_binary_path = os.path.join(folder_name, os.path.basename(binary_path))
    shutil.copyfile(binary_path, dest_binary_path)
    os.chmod(dest_binary_path, 0o755)
    
    subprocess.run(["./occamy_chip.vlt"], cwd=folder_name, check=True)
    with open(os.path.join(folder_name, "uart_0_0.log"), "r") as log_file:
        xdma_copy_line = next((ln for ln in log_file if "The XDMA copy is finished in" in ln), "")
        xdma_copy_cycle_val = xdma_copy_line.split("in")[-1].split("cycles")[0].strip() if xdma_copy_line else "N/A"

        idma_multicast_1_line = next((ln for ln in log_file if "The IDMA copy to 1 dest is finished in" in ln), "")
        idma_multicast_1_val = idma_multicast_1_line.split("in")[-1].split("cycles")[0].strip() if idma_multicast_1_line else "N/A"
        xdma_multicast_1_line = next((ln for ln in log_file if "The XDMA copy to 1 dest is finished in" in ln), "")
        xdma_multicast_1_val = xdma_multicast_1_line.split("in")[-1].split("cycles")[0].strip() if xdma_multicast_1_line else "N/A"

        idma_multicast_2_line = next((ln for ln in log_file if "The IDMA copy to 2 dest is finished in" in ln), "")
        idma_multicast_2_val = idma_multicast_2_line.split("in")[-1].split("cycles")[0].strip() if idma_multicast_2_line else "N/A"
        xdma_multicast_2_line = next((ln for ln in log_file if "The XDMA copy to 2 dest is finished in" in ln), "")
        xdma_multicast_2_val = xdma_multicast_2_line.split("in")[-1].split("cycles")[0].strip() if xdma_multicast_2_line else "N/A"

        idma_multicast_3_line = next((ln for ln in log_file if "The IDMA copy to 3 dest is finished in" in ln), "")
        idma_multicast_3_val = idma_multicast_3_line.split("in")[-1].split("cycles")[0].strip() if idma_multicast_3_line else "N/A"
        xdma_multicast_3_line = next((ln for ln in log_file if "The XDMA copy to 3 dest is finished in" in ln), "")
        xdma_multicast_3_val = xdma_multicast_3_line.split("in")[-1].split("cycles")[0].strip() if xdma_multicast_3_line else "N/A"

        idma_multicast_4_line = next((ln for ln in log_file if "The IDMA copy to 4 dest is finished in" in ln), "")
        idma_multicast_4_val = idma_multicast_4_line.split("in")[-1].split("cycles")[0].strip() if idma_multicast_4_line else "N/A"
        xdma_multicast_4_line = next((ln for ln in log_file if "The XDMA copy to 4 dest is finished in" in ln), "")
        xdma_multicast_4_val = xdma_multicast_4_line.split("in")[-1].split("cycles")[0].strip() if xdma_multicast_4_line else "N/A"


    with open(result_csv_path, "a") as result_file:
        result_file.write(f"{size},{idma_multicast_1_val},{xdma_copy_cycle_val},{idma_multicast_1_val},{xdma_multicast_1_val},{idma_multicast_2_val},{xdma_multicast_2_val},{idma_multicast_3_val},{xdma_multicast_3_val},{idma_multicast_4_val},{xdma_multicast_4_val}\n")

    # 5. Delete the entire folder
    shutil.rmtree(folder_name)

# Use ThreadPoolExecutor to run configurations concurrently

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for size in sizes:
        # 1. Prepa re folder name based on the configuration
        folder_name = f"S_{size}"
        os.makedirs(folder_name, exist_ok=True)

        # 2. Modify the params.hjson values in place
        param_data["size"] = size

        with open(param_path, 'w') as f:
            hjson.dump(param_data, f)

        # 3b. Execute "make sw" and then "make apps" in the apps folder
        apps_path = os.path.join(hemaia_root_path, "target/sim_chip/apps")
        subprocess.run(["make", "sw", "-j"], cwd=hemaia_root_path, check=True)
        subprocess.run(["make", "apps"], cwd=apps_path, check=True)

        # 4. Copy the "snax-xdma-transpose" folder to the new folder
        source_dir = os.path.join(apps_path, "offload-snax-xdma-multicast")
        target_dir = os.path.join(folder_name, "app_chip_0_0")
        shutil.copytree(source_dir, target_dir)

        futures.append(executor.submit(process_configuration, folder_name, size))

    # Wait for all threads to complete
    for future in futures:
        future.result()

print("All configurations have been processed.")
