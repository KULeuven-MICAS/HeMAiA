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

cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../rtl/cfg/hemaia_ci.hjson")
cfg_path = os.path.normpath(cfg_path)

task_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task.yaml")
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

        host_app = attr_dict.get('host_app', None)
        device_app = attr_dict.get('device_app', None)

        if host_app is None:
            raise ValueError("host_app must be specified in the configuration.")
        # if device_app is None, we assume it is the same as host_app
        app_name = host_app + "-" + device_app if device_app else host_app
        app_name_list.append(app_name)
        shell_script = ["make", "apps"]
        shell_script += [f"CFG={cfg_path}"]
        shell_script += [f"HOST_APP={host_app}"]
        shell_script += [f"DEVICE_APP={device_app}"] if device_app else []
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
