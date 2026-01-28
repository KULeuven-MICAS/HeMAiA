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

cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../rtl/cfg/lru.hjson")
cfg_path = os.path.normpath(cfg_path)

task_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task_vlt.yaml")
task_path = os.path.normpath(task_path)

num_threads = multiprocessing.cpu_count()


def get_apps(task_path):
    # Load the existing params from params.hjson (so we only modify the needed keys)
    with open(task_path, 'r') as f:
        param_data = yaml.safe_load(f)

    app_name_list = []
    for app_entry in param_data.get("runs", []):
        ci_app_name = list(app_entry.keys())[0]
        attributes = app_entry[ci_app_name]
        attr_dict = {}
        for attr in attributes:
            attr_dict.update(attr)
        
        # Try both uppercase and lowercase
        host_app_type = attr_dict.get('HOST_APP_TYPE', attr_dict.get('host_app_type'))
        chip_type = attr_dict.get('CHIP_TYPE', attr_dict.get('chip_type'))
        workload = attr_dict.get('WORKLOAD', attr_dict.get('workload'))
        dev_app = attr_dict.get('DEV_APP', attr_dict.get('dev_app'))

        if host_app_type is None:
            raise ValueError("HOST_APP_TYPE must be specified in the configuration.")
        
        # Determine the APP name (this should match what common.mk does)
        if host_app_type == "host_only":
            app_name = str(workload)
        elif host_app_type == "offload_legacy":
            app_name = f"offload_legacy_{chip_type}"
        else:
            app_name = f"{host_app_type}_{chip_type}_{workload}"

        app_name_list.append(app_name)
        shell_script = ["make", "apps"]
        shell_script += [f"CFG={cfg_path}"]
        shell_script += [f"HOST_APP_TYPE={host_app_type}"]
        shell_script += [f"CHIP_TYPE={chip_type}"] if chip_type and chip_type != "None" else []
        shell_script += [f"WORKLOAD={workload}"] if workload and workload != "None" else []
        shell_script += [f"DEV_APP={dev_app}"] if dev_app and dev_app != "None" else []
        # Generate the app hex file
        subprocess.run(shell_script, check=True, cwd=sw_path)
    return app_name_list


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

    try:
        subprocess.run([dest_binary_file], cwd=folder_name, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Task {sw_name} failed: {e}")
        raise e
    else:
        print(f"Task {sw_name} passed.")
    finally:
        shutil.rmtree(dest_binary_path)


def main():
    app_name_list = get_apps(task_path)

    # Use ThreadPoolExecutor to run configurations concurrently

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        current_task_num = 0
        for app_name in app_name_list:
            futures.append(executor.submit(process_configuration, f"task_{current_task_num}", app_name))
            current_task_num += 1

        # Wait for all threads to complete
        for future in futures:
            future.result()

    print("All configurations have been processed.")


if __name__ == "__main__":
    main()
