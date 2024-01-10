import time
import datetime
import subprocess
import yaml

from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration


DATA_SIZE_BYTES = 1000 * 1000 * 100  # 100 MB
TIME_UNIT = 1  # seconds


def start_flink(num_task_managers: int = 1):
    start_flink_cluster()
    start_task_managers(num_task_managers)


def start_flink_cluster():
    # First shut down all existing taskmanagers
    print(" [Shutting down all existing taskmanagers.]")
    subprocess.run("./flink-1.18.0/bin/stop-cluster.sh", check=True)
    subprocess.run(["./flink-1.18.0/bin/taskmanager.sh", "stop-all"], check=True)
    subprocess.run(["./flink-1.18.0/bin/historyserver.sh", "stop"], check=True)

    # Initialize the standalone cluster
    print(" [Starting a standalone Flink cluster.]")
    subprocess.run("./flink-1.18.0/bin/start-cluster.sh", check=True)
    subprocess.run(["./flink-1.18.0/bin/historyserver.sh", "start"], check=True)


def start_task_managers(num_task_managers: int = 1):
    # We need (num_task_managers - 1) additional taskmanagers, because start-cluster already spawns one taskmanager.
    print(" [Starting task managers.]")
    for _ in range(num_task_managers - 1):
        subprocess.run(["./flink-1.18.0/bin/taskmanager.sh", "start"], check=True)


def read_flink_conf(file_path="./flink-1.18.0/conf/flink-conf.yaml"):
    with open(file_path, "r") as file:
        conf = yaml.safe_load(file)
        return conf


def run_experiment(num_task_managers: int = 1):
    start_flink(num_task_managers)

    conf = read_flink_conf()
    print(f"Initialized {num_task_managers} TaskManagers.")
    print(f"Each TaskManager has {conf['taskmanager.numberOfTaskSlots']} task slots.")
    print(
        f"Total number of task slots: {num_task_managers * conf['taskmanager.numberOfTaskSlots']}"
    )
    # Submit the flink job
    print("[Submitting the flink job.]")
    subprocess.run(
        ["./flink-1.18.0/bin/flink", "run", "-py", "code.py"],
        check=True,
    )


def main():
    run_experiment()


if __name__ == "__main__":
    main()
