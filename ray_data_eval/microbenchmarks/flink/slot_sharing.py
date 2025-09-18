import time
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration
from pyflink.datastream.functions import FlatMapFunction, RuntimeContext, ProcessFunction
import json

NUM_CPUS = 8

slot_sharing = False

# Set optimal Parallelism
# PRODUCER_PARALLELISM = 4
# CONSUMER_PARALLELISM = 4

# Slot Sharing
PRODUCER_PARALLELISM = 8
CONSUMER_PARALLELISM = 8

EXECUTION_MODE = "process"
MB = 1024 * 1024

NUM_TASKS = 16 * 5
BLOCK_SIZE = int(1 * MB)
TIME_UNIT = 0.5

NUM_ROWS_PER_PRODUCER = 1
NUM_ROWS_PER_CONSUMER = 1


def append_dict_to_file(data: dict, file_path: str):
    """
    Append a dictionary to a file as a JSON object.

    Args:
        data (dict): The dictionary to append.
        file_path (str): The path to the file.
    """
    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary.")

    # Open the file in append mode, creating it if it doesn't exist
    with open(file_path, "a") as file:
        file.write(json.dumps(data) + "\n")


class Producer(FlatMapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def flat_map(self, value):
        producer_start = time.time()
        # print("Producer", value)
        time.sleep(TIME_UNIT * 3)
        producer_end = time.time()
        log = {
            "cat": "producer:" + str(self.task_index),
            "name": "producer:" + str(self.task_index),
            "pid": "",  # Be overwritten by parse.py
            "tid": "",  # Be overwritten by parse.py
            "ts": f"{producer_start * 1e6:.0f}",  # time is in microseconds
            "dur": f"{producer_end * 1e6 - producer_start * 1e6:.0f}",
            "ph": "X",
            "args": {},
        }
        # print(log)
        append_dict_to_file(log, "flink_logs.log")
        # logging.warning(json.dumps(log))

        for _ in range(NUM_ROWS_PER_PRODUCER):
            yield b"1" * BLOCK_SIZE


class ConsumerActor(ProcessFunction):
    current_batch = []
    idx = 0

    def open(self, runtime_context):
        self.current_batch = []
        self.idx = 0
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def process_element(self, value, _runtime_context):
        if len(self.current_batch) == 0:
            self.consumer_start = time.time()

        time.sleep(TIME_UNIT)
        self.current_batch.append(value)
        self.idx += 1
        if len(self.current_batch) < NUM_ROWS_PER_CONSUMER:
            return []

        self.consumer_end = time.time()
        log = {
            "cat": "consumer:" + str(self.task_index),
            "name": "consumer:" + str(self.task_index),
            "pid": "",  # Be overwritten by parse.py
            "tid": "",  # Be overwritten by parse.py
            "ts": f"{self.consumer_start * 1e6:.0f}",  # time is in microseconds
            "dur": f"{self.consumer_end * 1e6 - self.consumer_start * 1e6:.0f}",
            "ph": "X",
            "args": {},
        }
        append_dict_to_file(log, "flink_logs.log")

        # logging.warning(json.dumps(log))
        current_batch_len = len(self.current_batch)
        self.current_batch = []
        return [current_batch_len]


def run_flink(env):
    start = time.perf_counter()
    items = list(range(NUM_TASKS))
    ds = env.from_collection(items, type_info=Types.INT())
    producer = Producer()
    ds = (
        ds.flat_map(producer, output_type=Types.PICKLED_BYTE_ARRAY())
        .set_parallelism(PRODUCER_PARALLELISM)
        .slot_sharing_group("default")
    )

    ds = (ds.process(ConsumerActor()).set_parallelism(CONSUMER_PARALLELISM)).slot_sharing_group(
        "default"
    )

    result = []
    for length in ds.execute_and_collect():
        result.append(length)
        print(f"Processed block of size: {sum(result)}/{NUM_ROWS_PER_PRODUCER * len(items)}")

    total_length = sum(result)
    end = time.perf_counter()
    print(f"\nTotal data length: {total_length:,}")
    print(f"Time: {end - start:.4f}s")


def run_experiment():
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)
    config.set_integer("taskmanager.numberOfTaskSlots", NUM_CPUS)

    env = StreamExecutionEnvironment.get_execution_environment(config)
    run_flink(env)


def main():
    run_experiment()


if __name__ == "__main__":
    main()
