import time
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration
from pyflink.datastream.functions import FlatMapFunction, RuntimeContext, MapFunction
import argparse

EXECUTION_MODE = "process"
MB = 1024 * 1024
GB = 1024 * MB
TIME_UNIT = 0.5
NUM_CPUS = 8
NUM_GPUS = 4
NUM_ROWS_PER_TASK = 10
NUM_TASKS = 16 * 5
NUM_ROWS_TOTAL = NUM_ROWS_PER_TASK * NUM_TASKS
ROW_SIZE = 100 * MB
TIME_UNIT = 0.5


class Producer(FlatMapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def flat_map(self, value):
        time.sleep(TIME_UNIT * 10)
        for _ in range(NUM_ROWS_PER_TASK):
            yield b"1" * ROW_SIZE


class Consumer(MapFunction):
    def open(self, runtime_context):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def map(self, value):
        time.sleep(TIME_UNIT)
        return b"2" * ROW_SIZE


class Inference(MapFunction):
    def open(self, runtime_context):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def map(self, value):
        time.sleep(TIME_UNIT)
        return 1


def run_flink(env):
    start = time.perf_counter()
    items = list(range(NUM_TASKS))
    ds = env.from_collection(items, type_info=Types.INT())

    producer = Producer()
    ds = ds.flat_map(producer, output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(
        NUM_CPUS // 2
    )

    ds = ds.map(Consumer(), output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(NUM_CPUS // 2)

    ds = ds.map(Inference(), output_type=Types.LONG()).set_parallelism(NUM_GPUS)

    result = []
    for length in ds.execute_and_collect():
        result.append(length)
        print(f"Processed block of size: {length}, {sum(result)}")

    total_length = sum(result)

    end = time.perf_counter()
    print(f"\nTotal data length: {total_length:,}")
    print(f"Time: {end - start:.4f}s")


def run_experiment(mem_limit):
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)

    # Set memory limit for the task manager
    mem_limit_mb = mem_limit * 1024  # Convert GB to MB
    config.set_string("taskmanager.memory.process.size", f"{mem_limit_mb}m")

    env = StreamExecutionEnvironment.get_execution_environment(config)
    run_flink(env)


def main(mem_limit):
    run_experiment(mem_limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=20
    )
    args = parser.parse_args()

    main(args.mem_limit)
