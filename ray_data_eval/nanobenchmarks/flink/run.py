import subprocess
import yaml
import time

NUM_TASK_MANAGERS = 5
NUM_TASK_SLOTS = 1  # modify `taskmanager.numberOfTaskSlots` manually in flink-conf.yaml
FLINK_PATH = "../flink-1.18.0/"


def modify_workers_file(num_task_managers: int = 1):
    with open(FLINK_PATH + "conf/workers", "w") as file:
        file.writelines(["localhost\n"] * num_task_managers)


def read_flink_conf_file():
    flink_conf_path = FLINK_PATH + "conf/flink-conf.yaml"
    with open(flink_conf_path) as file:
        data = yaml.safe_load(file)
    return data


def start_flink(num_task_managers: int = 1):
    # First shut downt the existing cluster and taskmanagers
    print(" [Shutting down all existing taskmanagers.]")
    subprocess.run([FLINK_PATH + "bin/taskmanager.sh", "stop-all"], check=True)
    subprocess.run([FLINK_PATH + "bin/stop-cluster.sh"], check=True)
    subprocess.run([FLINK_PATH + "bin/historyserver.sh", "stop"], check=True)
    time.sleep(1)

    # Initialize the standalone cluster
    # By modifying the workers file, we initialize the correct number of taskmanagers
    print(" [Starting a standalone Flink cluster.]")
    modify_workers_file(num_task_managers)
    time.sleep(1)

    subprocess.run([FLINK_PATH + "bin/start-cluster.sh"], check=True)
    subprocess.run([FLINK_PATH + "bin/historyserver.sh", "start"], check=True)
    time.sleep(1)


def run_experiment(num_task_managers: int = 1):
    start_flink(num_task_managers)

    conf = read_flink_conf_file()
    print(f"Initialized {num_task_managers} TaskManagers.")
    print(f"Each TaskManager has {conf['taskmanager.numberOfTaskSlots']} task slots.")
    print(
        f"Total number of task slots: {num_task_managers * conf['taskmanager.numberOfTaskSlots']}"
    )
    # Submit the flink job
    print(" [Submitting the flink job.]")
    time.sleep(1)
    subprocess.run(
        [FLINK_PATH + "bin/flink", "run", "-py", "./code.py"],
        check=True,
    )


def main():
    run_experiment(num_task_managers=NUM_TASK_MANAGERS)


if __name__ == "__main__":
    main()
