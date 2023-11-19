import time

import numpy as np
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment

DATA_SIZE = 1000 * 100

env = StreamExecutionEnvironment.get_execution_environment()


def memory_blowup(x, blowup: int):
    return [x + np.random.rand(DATA_SIZE).tolist() for _ in range(blowup)]


# Experiment function
def run_experiment(env, blowup: int = -1, parallelism: int = -1, size: int = -1):
    start = time.perf_counter()

    ds = env.from_collection(range(size), type_info=Types.INT())
    ds = ds.map(lambda _: np.random.rand(DATA_SIZE), output_type=Types.PICKLED_BYTE_ARRAY())
    ds = ds.map(lambda array: array.nbytes, output_type=Types.LONG())
    ds = ds.key_by(lambda _: 1).reduce(lambda a, b: a + b)

    with ds.execute_and_collect() as ret:
        ret = list(ret)
        print(f"{ret[-1]:,}")

    end = time.perf_counter()
    print(f"Time: {end - start:.4f}s")


def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(50)

    run_experiment(env, size=10000)


if __name__ == "__main__":
    main()
