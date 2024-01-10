import time
import datetime
import subprocess

from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration

from ray_data_eval.common.types import SchedulingProblem

DATA_SIZE_BYTES = 1000 * 1000 * 100  # 100 MB
TIME_UNIT = 1  # seconds


def start_flink(cfg: SchedulingProblem):
    start_flink_cluster()
    start_task_managers(cfg.num_producers)


def start_flink_cluster():
    # Shutdown all existing taskmanagers
    subprocess.run(["./bin/taskmanager.sh", "stop-all"], check=True)
    subprocess.run("./bin/stop-cluster.sh", check=True)
    subprocess.run(["./bin/flink/historysever.sh", "stop"], check=True)

    subprocess.run("./bin/flink/start-cluster.sh", check=True)
    subprocess.run(["./bin/flink/historysever.sh", "start"], check=True)


def start_task_managers(num_task_managers: int = 0):
    # We need (cfg.num_execution_slots - 1) additional taskmanagers, because start-cluster already spawns one taskmanager.
    for _ in range(num_task_managers - 1):
        subprocess.run(["./bin/taskmanager.sh", "start"], check=True)


def producer(item, cfg: SchedulingProblem):
    data = b"1" * (DATA_SIZE_BYTES * cfg.producer_output_size[item])
    time.sleep(TIME_UNIT * cfg.producer_time[item])
    return (data, item)


def consumer(item, cfg: SchedulingProblem):
    data, i = item
    time.sleep(TIME_UNIT * cfg.consumer_time[i])
    return len(data)


def run_flink(env, cfg: SchedulingProblem):
    if cfg.num_producers != cfg.num_consumers:
        raise NotImplementedError(f"num_producers != num_consumers: {cfg}")

    start = time.perf_counter()

    items = list(range(cfg.num_producers))
    ds = env.from_collection(items, type_info=Types.INT())

    ds = ds.map(
        lambda x: producer(x, cfg),
        output_type=Types.TUPLE([Types.PICKLED_BYTE_ARRAY(), Types.INT()]),
    ).disable_chaining()

    ds = ds.map(lambda x: consumer(x, cfg), output_type=Types.LONG())

    result = ds.execute_and_collect()
    total_length = sum(result)

    end = time.perf_counter()
    print(f"\nTotal data length: {total_length:,}")
    print(f"Time: {end - start:.4f}s")


def run_experiment(cfg: SchedulingProblem):
    # wandb.init(project="ray-data-eval-flink", entity="raysort")
    # wandb.config.update(cfg)M

    start_flink(cfg)
    run_flink(cfg)


def main():
    run_experiment(
        SchedulingProblem(
            num_producers=5,
            num_consumers=5,
            # producer_time=3,
            consumer_time=2,
            # producer_output_size=2,
            # consumer_input_size=2,
            time_limit=20,
            num_execution_slots=1,
            buffer_size_limit=1,
        ),
    )


if __name__ == "__main__":
    main()
