import os
import subprocess
import shutil
import yaml
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

binary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../bin/occamy_chip.vlt")
binary_path = os.path.normpath(binary_path)

sw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../apps")
sw_path = os.path.normpath(sw_path)

cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task.yaml")
cfg_path = os.path.normpath(cfg_path)

num_threads = multiprocessing.cpu_count()

# Load the existing params from params.hjson (so we only modify the needed keys)
with open(cfg_path, 'r') as f:
    param_data = yaml.safe_load(f)

# The function needs to be parallelized
def process_configuration(folder_name, sw_name):
    # 3a. Copy hardware binary into the new folder
    dest_binary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name)
    os.makedirs(dest_binary_path, exist_ok=True)
    dest_binary_file = os.path.join(dest_binary_path, "occamy_chip.vlt")
    shutil.copyfile(binary_path, dest_binary_file)
    os.chmod(dest_binary_file, 0o755)
    src_sw_name = os.path.join(sw_path, sw_name)
    dest_sw_dir = os.path.join(dest_binary_path, "app_chip_0_0")
    shutil.copytree(src_sw_name, dest_sw_dir)

    subprocess.run([dest_binary_file], cwd=folder_name, check=True)
    shutil.rmtree(dest_binary_path)

# Use ThreadPoolExecutor to run configurations concurrently

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    current_task_num = 0
    for run_cfg in param_data.get("runs", []):
        elf_path = run_cfg.get("elf")
        if elf_path:
            # You can use elf_path as needed in your configuration loop
            # For example, pass it to process_configuration or use it to set up the folder
            futures.append(executor.submit(process_configuration, f"task_{current_task_num}", elf_path))
            current_task_num += 1

    # Wait for all threads to complete
    for future in futures:
        future.result()

print("All configurations have been processed.")
