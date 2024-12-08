import time
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration, WatermarkStrategy
from pyflink.datastream.connectors.number_seq import NumberSequenceSource
from pyflink.datastream.functions import FlatMapFunction, ProcessFunction
import json
from pyflink.datastream import (
    CheckpointingMode
)
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import RuntimeContext

NUM_CPUS = 8

# Slot Sharing
PRODUCER_PARALLELISM = 8
CONSUMER_PARALLELISM = 8

EXECUTION_MODE = "process"
MB = 1024 * 1024    

NUM_TASKS = 16 * 5 * 100
BLOCK_SIZE = int(1 * MB)
TIME_UNIT = 0.1

NUM_ROWS_PER_PRODUCER = 1
NUM_ROWS_PER_CONSUMER = 1

start_time = time.time()

def busy_loop(duration_in_seconds):
    """
    A busy loop that runs for the specified duration in seconds.

    :param duration_in_seconds: The time (in seconds) the loop should run.
    """
    end_time = time.time() + duration_in_seconds
    while time.time() < end_time:
        pass  # Busy waiting

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

class Producer(FlatMapFunction):
    def __init__(self):
        self.count = None  # Will be backed by distributed state (MapState)

    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()
        self.attempt_number = runtime_context.get_attempt_number()
        print(f"Opened Producer {self.task_info} with index {self.task_index}")

        # Define a MapState descriptor and initialize the state (for distributed state)
        state_descriptor = ValueStateDescriptor("count_state", Types.INT())
        self.count = runtime_context.get_state(state_descriptor)
        print(f"Init state for task {self.task_index}")

    def flat_map(self, value):

        self.count.update((self.count.value() or 0) + 1)

        producer_start = time.time()
        busy_loop(TIME_UNIT * 3)
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
        for i in range(NUM_ROWS_PER_PRODUCER):
            yield i + value * NUM_ROWS_PER_PRODUCER

        # Simulate failure after a certain count
        print(f"count: {self.count.value()}, value: {value}")
        if self.count.value() == NUM_TASKS // 20 and self.attempt_number == 0:
            print(f"Injecting failure {time.time() - start_time:.2f}s since start: Attempt {self.attempt_number}")
            raise RuntimeError("Simulated failure in Producer")

        
        
class ConsumerActor(ProcessFunction):

    def open(self, runtime_context):
        self.current_batch = []
        self.idx = 0
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def process_element(self, value, _runtime_context):
        if len(self.current_batch) == 0:
            self.consumer_start = time.time()

        busy_loop(TIME_UNIT)
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

        batch = self.current_batch
        self.current_batch = []
        return self.current_batch

def run_flink(env):
    
    start = time.perf_counter()
    number_source = NumberSequenceSource(0, NUM_TASKS)

    ds = env.from_source(
        source=number_source,
        watermark_strategy=WatermarkStrategy.for_monotonous_timestamps(),
        source_name="file_source",
        type_info=Types.LONG(),
    ).set_parallelism(1)
        
    # Apply a key_by transformation to create a KeyedStream
    ds = ds.key_by(lambda x: x % PRODUCER_PARALLELISM)
    ds = ds.flat_map(Producer(), output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(PRODUCER_PARALLELISM) 
    ds = ds.process(ConsumerActor()).set_parallelism(CONSUMER_PARALLELISM)
    ds.print()

    # Execute the Flink job
    env.execute("Streaming Job Example")
    end = time.perf_counter()
    print(f"Time: {end - start:.4f}s")

def run_experiment():
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)
    config.set_integer("taskmanager.numberOfTaskSlots", NUM_CPUS)
    
    config.set_string('restart-strategy.type', 'fixed-delay')
    config.set_string('restart-strategy.fixed-delay.attempts', '3') # number of restart attempts
    config.set_string('restart-strategy.fixed-delay.delay', '3000ms') # delay
    
    config.set_string("state.backend.type", "rocksdb")  # Use a persistent backend
    config.set_string("state.checkpoints.dir", "file:///home/ubuntu/ray-data-eval/ray_data_eval/microbenchmarks/flink/flink-checkpoints")  # Local or remote path
    # config.set_boolean("state.backend.incremental", True)  # Enable incremental checkpointing

    env = StreamExecutionEnvironment.get_execution_environment(config)
    env.enable_checkpointing(30000, CheckpointingMode.EXACTLY_ONCE)
    print("checkpoint enabled: ", env.get_checkpoint_config().is_checkpointing_enabled())
    print("interval: ", env.get_checkpoint_config().get_checkpoint_interval())
    run_flink(env)

def main():
    run_experiment()


if __name__ == "__main__":
    main()
