import time
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration
from pyflink.datastream.functions import FlatMapFunction, ProcessFunction
import logging
import json
import os
from pyflink.datastream import (
    CheckpointingMode
)
from pyflink.datastream.state import ValueStateDescriptor, ValueState, MapStateDescriptor, MapState
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import RuntimeContext

NUM_CPUS = 8

# Slot Sharing
PRODUCER_PARALLELISM = 1
CONSUMER_PARALLELISM = 1

EXECUTION_MODE = "process"
MB = 1024 * 1024    

NUM_TASKS = 16 * 5 * 10
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
    with open(file_path, 'a') as file:
        file.write(json.dumps(data) + '\n')
        
# class CheckpointLoggingListener(CheckpointListener):
#     def notify_checkpoint_completed(self, checkpoint_id: int) -> None:
#         print(f"Checkpoint {checkpoint_id} completed successfully.")

#     def notify_checkpoint_failed(self, checkpoint_id: int) -> None:
#         print(f"Checkpoint {checkpoint_id} failed.")
        
class Producer(FlatMapFunction):
    def __init__(self):
        self.count = None  # Will be backed by distributed state (MapState)

    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()
        print(f"Opened Producer {self.task_info} with index {self.task_index}")

        # Define a MapState descriptor and initialize the state (for distributed state)
        state_descriptor = MapStateDescriptor("count_state", Types.INT(), Types.INT())
        self.count: MapState = runtime_context.get_map_state(state_descriptor)
        print(f"Restored state for task {self.task_index}: {self.count}, {state_descriptor.get_name()}, id(runtime_context): {id(runtime_context)}")

    def flat_map(self, value):

        # Increment count and simulate processing
        # Retrieve the current state or initialize it
        current_count = 0
        if self.count.contains(self.task_index):  # Ensure the key exists
            current_count = self.count.get(self.task_index)

        # Increment the state
        current_count += 1
        # Update the state
        self.count.put(self.task_index, current_count)
        
        producer_start = time.time()
        time.sleep(TIME_UNIT * 3)
        producer_end = time.time()

        # Log processing details
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
        append_dict_to_file(log, 'flink_logs.log')

        # Generate output
        for _ in range(NUM_ROWS_PER_PRODUCER):
            yield b"1" * BLOCK_SIZE

        # Simulate failure after a certain count
        print(f"count: {current_count}")
        if current_count == 8:
            print(f"Simulating failure in Producer, count: {current_count}")
            raise RuntimeError("Simulated failure in Producer")

        
        
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
        append_dict_to_file(log, 'flink_logs.log')

        # logging.warning(json.dumps(log))
        current_batch_len = len(self.current_batch)
        self.current_batch = []
        return [current_batch_len]

def configure_restart_strategy(config):
    config.set_string('restart-strategy.type', 'fixed-delay')
    config.set_string('restart-strategy.fixed-delay.attempts', '3') # number of restart attempts
    config.set_string('restart-strategy.fixed-delay.delay', '1000ms') # delay

def run_flink(env):
    
    start = time.perf_counter()
    items = list(range(NUM_TASKS))
    ds = env.from_collection(items, type_info=Types.INT())
    
    # Apply a key_by transformation to create a KeyedStream
    ds = ds.key_by(lambda x: x % PRODUCER_PARALLELISM)

    producer = Producer()
    ds = ds.flat_map(producer, output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(
        PRODUCER_PARALLELISM
    ) 
    
    ds = (
        ds.process(ConsumerActor()).set_parallelism(CONSUMER_PARALLELISM)
    )
    
    # result = []
    # for length in ds.execute_and_collect():
    #     result.append(length)
    #     print(f"Processed block of size: {sum(result)}/{NUM_ROWS_PER_PRODUCER * len(items)}", flush=True)

    # Use a print sink to observe the output
    ds.print()

    # Execute the Flink job
    env.execute("Streaming Job Example")

    # total_length = sum(result)
    end = time.perf_counter()
    # print(f"\nTotal data length: {total_length:,}")
    print(f"Time: {end - start:.4f}s")

def run_experiment():
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)
    config.set_integer("taskmanager.numberOfTaskSlots", NUM_CPUS)
    
    configure_restart_strategy(config)

    config.set_string("state.backend", "filesystem")  # Use a persistent backend
    config.set_string("state.checkpoints.dir", "file:///home/ubuntu/ray-data-eval/ray_data_eval/microbenchmarks/flink/flink-checkpoints")  # Local or remote path


    env = StreamExecutionEnvironment.get_execution_environment(config)
    env.enable_checkpointing(1000)
    env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    run_flink(env)

def main():
    run_experiment()


if __name__ == "__main__":
    main()
