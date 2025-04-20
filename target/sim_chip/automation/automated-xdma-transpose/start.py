import os
import subprocess
import shutil
import hjson
from concurrent.futures import ThreadPoolExecutor

param_path = "/users/micas/ydeng/Desktop/SNAX/HeMAiA/target/sim/sw/device/apps/snax/snax-xdma-transpose/data/params.hjson"
binary_path = "/users/micas/ydeng/Desktop/SNAX/HeMAiA/target/sim_chip/bin/occamy_chip.vlt"
hemaia_root_path = "/users/micas/ydeng/Desktop/SNAX/HeMAiA"

num_threads = 12
Ms = [32, 64, 96, 128, 192, 256, 384, 512]
Ns = [32, 64, 96, 128, 192, 256, 384, 512]
BIT_WIDTHs = [8, 16]
input_layouts = ["MNM8N8", "MN"]
output_layouts = ["MNM8N8", "MN"]
transpose_options = [True]

# Load the existing params from params.hjson (so we only modify the needed keys)
with open(param_path, 'r') as f:
    param_data = hjson.load(f)


# Create or overwrite the result.csv file
result_csv_path = "result.csv"
with open(result_csv_path, 'w') as result_csv:
    # Write the header row
    result_csv.write("M,N,BIT_WIDTH,input_layout,output_layout,enable_transpose,idma_transposer,xdma\n")

# The function needs to be parallelized
def process_configuration(folder_name, M, N, bit_width, input_layout, output_layout, transpose):
    # 3a. Copy hardware binary into the new folder
    dest_binary_path = os.path.join(folder_name, os.path.basename(binary_path))
    shutil.copyfile(binary_path, dest_binary_path)
    os.chmod(dest_binary_path, 0o755)
    
    subprocess.run(["./occamy_chip.vlt"], cwd=folder_name, check=True)
    with open(os.path.join(folder_name, "uart_0_0.log"), "r") as log_file:
        baseline_cycle_line = next((ln for ln in log_file if "The IDMA + Transposer copy is finished in" in ln), "")
        baseline_cycle_val = baseline_cycle_line.split("in")[-1].split("cycles")[0].strip() if baseline_cycle_line else "N/A"

        xdma_cycle_line = next((ln for ln in log_file if "The XDMA copy is finished in" in ln), "")
        xdma_cycle_val = xdma_cycle_line.split("in")[-1].split("cycles")[0].strip() if baseline_cycle_line else "N/A"

    with open(result_csv_path, "a") as result_file:
        result_file.write(f"{M},{N},{bit_width},{input_layout},{output_layout},{transpose},{baseline_cycle_val},{xdma_cycle_val}\n")

    # 5. Delete the entire folder
    shutil.rmtree(folder_name)

# Use ThreadPoolExecutor to run configurations concurrently

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for M in Ms:
        for N in Ns:
            for bit_width in BIT_WIDTHs:
                for input_layout in input_layouts:
                    for output_layout in output_layouts:
                        for transpose in transpose_options:
                            # 1. Prepa re folder name based on the configuration
                            folder_name = f"M_{M}_N_{N}_BW_{bit_width}_IL_{input_layout}_OL_{output_layout}_T_{transpose}"
                            os.makedirs(folder_name, exist_ok=True)

                            # 2. Modify the params.hjson values in place
                            param_data["M"] = M
                            param_data["N"] = N
                            param_data["BIT_WIDTH"] = bit_width
                            param_data["input_layout"] = input_layout
                            param_data["output_layout"] = output_layout
                            param_data["enable_transpose"] = transpose

                            with open(param_path, 'w') as f:
                                hjson.dump(param_data, f)

                            # 3b. Execute "make sw" and then "make apps" in the apps folder
                            apps_path = os.path.join(hemaia_root_path, "target/sim_chip/apps")
                            subprocess.run(["make", "sw", "-j"], cwd=hemaia_root_path, check=True)
                            subprocess.run(["make", "apps"], cwd=apps_path, check=True)

                            # 4. Copy the "snax-xdma-transpose" folder to the new folder
                            source_dir = os.path.join(apps_path, "offload-snax-xdma-transpose")
                            target_dir = os.path.join(folder_name, "app_chip_0_0")
                            shutil.copytree(source_dir, target_dir)

                            futures.append(executor.submit(process_configuration, folder_name, M, N, bit_width, input_layout, output_layout, transpose))

    # Wait for all threads to complete
    for future in futures:
        future.result()

print("All configurations have been processed.")
