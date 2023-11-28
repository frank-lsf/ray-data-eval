import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np

DATA_SIZE = 1000 * 100  # 100 KB


def gen_data(_):
    return (np.random.rand(DATA_SIZE),)


def memory_blowup(x, *, blowup: int):
    return np.concatenate([x + np.random.rand(DATA_SIZE) for _ in range(blowup)])


def sum_byte_sizes(a, b):
    size_a = a if isinstance(a, int) else a[0].nbytes
    size_b = b if isinstance(b, int) else b[0].nbytes
    return size_a + size_b


def run_experiment(spark, blowup: int = 0, parallelism: int = -1, size: int = -1):
    start = time.perf_counter()

    rdd = spark.sparkContext.parallelize(range(size), parallelism)
    rdd = rdd.map(gen_data)
    if blowup > 0:
        rdd = rdd.map(lambda x: memory_blowup(x, blowup=blowup))
    ret = rdd.reduce(sum_byte_sizes)

    end = time.perf_counter()
    print(f"\n{ret:,}")
    print(f"Time: {end - start:.4f}s")
    return ret


def main():
    spark = (
        SparkSession.builder.appName("Spark")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .getOrCreate()
    )

    run_experiment(spark, parallelism=100, size=10000, blowup=20)


if __name__ == "__main__":
    main()
