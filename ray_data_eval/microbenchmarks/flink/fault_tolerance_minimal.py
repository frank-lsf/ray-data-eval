import time
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration
from pyflink.datastream.functions import FlatMapFunction
from pyflink.datastream import (
    CheckpointingMode
)
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor

# Slot Sharing
PRODUCER_PARALLELISM = 1
CONSUMER_PARALLELISM = 1

NUM_TASKS = 16 * 5 * 10000
TIME_UNIT = 0.5

class Producer(FlatMapFunction):
    def __init__(self):
        self.count = None  

    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()
        
        descriptor = ValueStateDescriptor(
            "average",  # the state name
            Types.INT()  # Use INT type for integer values
        )
        self.count = runtime_context.get_state(descriptor)
        print(f"Opened Producer {self.task_info} with index {self.task_index}")

    def flat_map(self, value):
        self.count.update(self.count.value() or 0 + 1)  
        print(f"count: {self.count.value()}, {value}")
        cnt = 0
        while cnt < 10000:
            cnt += 1
        yield value
        
        # if self.count == NUM_TASKS // 20:
            # raise RuntimeError("Failure injected")

def run_flink(env):
    
    start = time.perf_counter()
    items = list(range(NUM_TASKS))
    ds = env.from_collection(items, type_info=Types.INT())
    
    # Apply a key_by transformation to create a KeyedStream
    ds = ds.key_by(lambda x: x % PRODUCER_PARALLELISM)
    ds = ds.flat_map(Producer(), output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(
        PRODUCER_PARALLELISM
    ) 
    
    # Execute the Flink job
    env.execute("Streaming Job Example")

    # total_length = sum(result)
    end = time.perf_counter()
    # print(f"\nTotal data length: {total_length:,}")
    print(f"Time: {end - start:.4f}s")

def run_experiment():
    config = Configuration()
    config.set_string('restart-strategy.type', 'fixed-delay')
    config.set_string('restart-strategy.fixed-delay.attempts', '3') # number of restart attempts
    config.set_string('restart-strategy.fixed-delay.delay', '10000ms') # delay
    config.set_string("state.backend.type", "rocksdb")  # Use a persistent backend
    config.set_string("state.checkpoints.dir", "file:///home/ubuntu/ray-data-eval/ray_data_eval/microbenchmarks/flink/flink-checkpoints")  # Local or remote path


    env = StreamExecutionEnvironment.get_execution_environment(config)
    env = env.enable_checkpointing(5000, CheckpointingMode.EXACTLY_ONCE)
    print("checkpoint enabled: ", env.get_checkpoint_config().is_checkpointing_enabled())
    print("interval: ", env.get_checkpoint_config().get_checkpoint_interval())
    run_flink(env)

def main():
    run_experiment()


if __name__ == "__main__":
    main()
