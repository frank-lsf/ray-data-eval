import time
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration
from pyflink.datastream.functions import FlatMapFunction, RuntimeContext, MapFunction
import argparse
import resource
import json

from setting import (
    GB,
    EXECUTION_MODE,
    TIME_UNIT,
    NUM_CPUS,
    NUM_GPUS,
    FRAMES_PER_VIDEO,
    NUM_VIDEOS,
    NUM_FRAMES_TOTAL,
    FRAME_SIZE_B,
    append_dict_to_file,
    log_memory_usage_process,
    limit_cpu_memory
)

class Producer(FlatMapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def flat_map(self, value):
        
        producer_start = time.time()
        time.sleep(TIME_UNIT * 10)
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
        append_dict_to_file(log, 'flink_logs_producer_consumer_gpu.log')

        for _ in range(FRAMES_PER_VIDEO):
            yield b"1" * FRAME_SIZE_B


class Consumer(MapFunction):
    def open(self, runtime_context):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def map(self, value):
        consumer_start = time.time()
        time.sleep(TIME_UNIT)
        consumer_end = time.time()

        # Log processing details
        log = {
            "cat": "consumer:" + str(self.task_index),
            "name": "consumer:" + str(self.task_index),
            "pid": "",  # Be overwritten by parse.py
            "tid": "",  # Be overwritten by parse.py
            "ts": f"{consumer_start * 1e6:.0f}",  # time is in microseconds
            "dur": f"{consumer_end * 1e6 - consumer_start * 1e6:.0f}",
            "ph": "X",
            "args": {},
        }
        append_dict_to_file(log, 'flink_logs_producer_consumer_gpu.log')

        return b"2" * FRAME_SIZE_B


class Inference(MapFunction):
    def open(self, runtime_context):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def map(self, value):
        inference_start = time.time()
        time.sleep(TIME_UNIT)
        inference_end = time.time()

        # Log processing details
        log = {
            "cat": "inference:" + str(self.task_index),
            "name": "inference:" + str(self.task_index),
            "pid": "",  # Be overwritten by parse.py
            "tid": "",  # Be overwritten by parse.py
            "ts": f"{inference_start * 1e6:.0f}",  # time is in microseconds
            "dur": f"{inference_end * 1e6 - inference_start * 1e6:.0f}",
            "ph": "X",
            "args": {},
        }
        append_dict_to_file(log, 'flink_logs_producer_consumer_gpu.log')
        return 1


def run_flink(env, mem_limit):
    start = time.perf_counter()
    items = list(range(NUM_VIDEOS))
    ds = env.from_collection(items, type_info=Types.INT())

    producer = Producer()


    if mem_limit <= 8:
        p = 1
    else:
        p =  NUM_CPUS // 2
        
        
    ds = ds.flat_map(producer, output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(
        p
    )

    ds = ds.map(Consumer(), output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(
        p
    )

    ds = ds.map(Inference(), output_type=Types.LONG()).set_parallelism(
        NUM_GPUS if mem_limit > 8 else 1
    )

    count = 0
    for length in ds.execute_and_collect():
        count += length
        print(f"Processed {count}/{NUM_FRAMES_TOTAL}")

    end = time.perf_counter()
    print(f"Total rows: {count:,}")
    print(f"Run time: {end - start:.4f}s")


def run_experiment(mem_limit):
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)

    # Set memory limit for the task manager
    # https://nightlies.apache.org/flink/flink-docs-master/docs/deployment/memory/mem_setup/
    mem_limit_mb = mem_limit * 1024  # Convert GB to MB
    print("memory: ", f"{mem_limit_mb / NUM_CPUS}m")
    config.set_string("taskmanager.memory.process.size", f"{mem_limit_mb / NUM_CPUS}m")
    env = StreamExecutionEnvironment.get_execution_environment(config)
    run_flink(env, mem_limit)


def main(mem_limit):
    run_experiment(mem_limit)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=20
    )
    args = parser.parse_args()

    # import multiprocessing
    # Start memory usage logging in a separate process
    # logging_process = multiprocessing.Process(target=log_memory_usage_process, args=(2, args.mem_limit))  # Log every 2 seconds
    # logging_process.start()
    limit_cpu_memory(args.mem_limit)

    main(args.mem_limit)
    # logging_process.terminate()
