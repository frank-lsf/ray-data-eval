import time
import sys

from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration

from ray_data_eval.common.types import SchedulingProblem, test_problem

DATA_SIZE_BYTES = 1000 * 1000 * 100  # 100 MB
TIME_UNIT = 1  # seconds


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
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(cfg.num_producers)

    run_flink(env, cfg)


def main():
    run_experiment(test_problem)


if __name__ == "__main__":
    main()
