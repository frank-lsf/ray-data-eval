import time
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment, MapFunction
from pyflink.common import Configuration


# # # # # # # # # # #

NUM_CPUS = 8
PRODUCER_PARALLELISM = 2
CONSUMER_PARALLELISM = NUM_CPUS - PRODUCER_PARALLELISM
EXECUTION_MODE = "process"
MB = 1024 * 1024

NUM_ROWS_PER_TASK = 10
NUM_TASKS = 16 * 5
NUM_ROWS_TOTAL = NUM_ROWS_PER_TASK * NUM_TASKS
BLOCK_SIZE = 1 * MB # TODO(Bug. 100MB doesn't work.)
TIME_UNIT = 0.1


class Producer(MapFunction):
    def map(self, item):
        data = [b"1" * BLOCK_SIZE for _ in range(10)]
        print("Producer", item)
        time.sleep(TIME_UNIT * 10)
        return (data, item)


class Consumer(MapFunction):
    def map(self, item):
        data, i = item
        print("Consumer", i)
        time.sleep(TIME_UNIT)
        return len(data)


def run_flink(env):
    start = time.perf_counter()

    items = list(range(NUM_ROWS_TOTAL))
    ds = env.from_collection(items, type_info=Types.INT())

    producer = Producer()
    consumer = Consumer()

    ds = (
        ds.map(producer, output_type=Types.TUPLE([Types.PICKLED_BYTE_ARRAY(), Types.INT()]))
        .set_parallelism(PRODUCER_PARALLELISM)
        .rebalance()  # Rebalance to distribute data evenly
        .flat_map(
            lambda blocks: ((block, blocks[1]) for block in blocks[0]),
            output_type=Types.TUPLE([Types.PICKLED_BYTE_ARRAY(), Types.INT()]),
        )
        .disable_chaining()
    )

    ds = (
        ds.map(consumer, output_type=Types.LONG())
        .set_parallelism(CONSUMER_PARALLELISM)
        .disable_chaining()
    )

    # Collecting results in a list to avoid overwhelming the collection with huge data.
    result = []
    for length in ds.execute_and_collect():
        result.append(length)
        print(f"Processed block of size: {length}")

    total_length = sum(result)

    end = time.perf_counter()
    print(f"\nTotal data length: {total_length:,}")
    print(f"Time: {end - start:.4f}s")


def run_experiment():
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)
    config.set_string("taskmanager.memory.process.size", "2g")
    config.set_string("jobmanager.memory.process.size", "2g")
    env = StreamExecutionEnvironment.get_execution_environment(config)
    run_flink(env)


def main():
    run_experiment()


if __name__ == "__main__":
    main()
