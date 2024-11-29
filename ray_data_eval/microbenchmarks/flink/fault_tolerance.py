import time
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration
from pyflink.datastream.functions import FlatMapFunction, RuntimeContext, ProcessFunction
import logging
import json
import os
from pyflink.datastream import (
    CheckpointingMode
)
from pyflink.datastream.state import ValueStateDescriptor


NUM_CPUS = 8

# Slot Sharing
PRODUCER_PARALLELISM = 8
CONSUMER_PARALLELISM = 8

EXECUTION_MODE = "process"
MB = 1024 * 1024    

NUM_TASKS = 16 * 5 * 10
BLOCK_SIZE = int(1 * MB)
TIME_UNIT = 0.5

NUM_ROWS_PER_PRODUCER = 1
NUM_ROWS_PER_CONSUMER = 1

failure_injected = False

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
        
    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

        # Define state descriptor for self.count
        # self.count_state_descriptor = ValueStateDescriptor("count_state", Types.INT())
        
        # # Initialize state
        # self.count_state = runtime_context.get_state(self.count_state_descriptor)
        
        # # Try to restore the state if it exists
        # self.count = self.count_state.value() if self.count_state.value() is not None else 0
        self.count = 0        

    def flat_map(self, value):
        self.count += 1
        global failure_injected
        
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
        append_dict_to_file(log, 'flink_logs.log')
        # logging.warning(json.dumps(log))

        for _ in range(NUM_ROWS_PER_PRODUCER):
            yield b"1" * BLOCK_SIZE

        print(f"self.count: {self.count}, failure_injected: {failure_injected}, {id(failure_injected)}")
        if not failure_injected and self.count == 8:  # Simulate failure after processing 5000 elements
            failure_injected = True
            print(f"failure_injected: {failure_injected}, {id(failure_injected)}")
            # raise RuntimeError("Simulated failure in Producer")

        # Update state with the current count
        # self.count_state.update(self.count)
        
        
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

def configure_checkpointing(config):
    # Set checkpointing interval and mode
    config.set_string('execution.checkpointing.interval', '1000')  # 1000 ms (1 second)
    config.set_string('execution.checkpointing.mode', 'EXACTLY_ONCE')  # Exactly once mode
    
def configure_restart_strategy(config):
    config.set_string('restart-strategy.type', 'fixed-delay')
    config.set_string('restart-strategy.fixed-delay.attempts', '3') # number of restart attempts
    config.set_string('restart-strategy.fixed-delay.delay', '10000 ms') # delay

def run_flink(env):
    
    start = time.perf_counter()
    items = list(range(NUM_TASKS))
    ds = env.from_collection(items, type_info=Types.INT())
    
    producer = Producer()
    ds = ds.flat_map(producer, output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(
        PRODUCER_PARALLELISM
    ).slot_sharing_group("default")    
    
    ds = (
        ds.process(ConsumerActor()).set_parallelism(CONSUMER_PARALLELISM)
    ).slot_sharing_group("default")
    
    result = []
    for length in ds.execute_and_collect():
        result.append(length)
        print(f"Processed block of size: {sum(result)}/{NUM_ROWS_PER_PRODUCER * len(items)}", flush=True)

    total_length = sum(result)
    end = time.perf_counter()
    print(f"\nTotal data length: {total_length:,}")
    print(f"Time: {end - start:.4f}s")

def run_experiment():
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)
    config.set_integer("taskmanager.numberOfTaskSlots", NUM_CPUS)
    configure_restart_strategy(config)
    configure_checkpointing(config)

    env = StreamExecutionEnvironment.get_execution_environment(config)
    run_flink(env)

def main():
    run_experiment()


if __name__ == "__main__":
    main()
