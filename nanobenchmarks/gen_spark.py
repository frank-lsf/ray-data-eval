from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np
import time

DATA_SIZE = 1000 * 100


def gen_data(_):
    return (np.random.rand(DATA_SIZE),)


def memory_blowup(x, *, blowup: int):
    return [x + np.random.rand(DATA_SIZE) for _ in range(blowup)]


def get_nbytes(row):
    return (row[0].nbytes,)


def run_experiment(spark, blowup: int = -1, parallelism: int = -1, size: int = -1):
    start = time.perf_counter()

    rdd = spark.sparkContext.parallelize(range(size), parallelism)
    rdd = rdd.map(gen_data)
    if blowup > 0:
        rdd = rdd.flatMap(lambda x: memory_blowup(x, blowup=blowup))
    rdd = rdd.map(get_nbytes)
    ret = rdd.reduce(lambda x, y: (x[0] + y[0],))[0]

    end = time.perf_counter()
    print(f"\n{ret:,}")
    print(f"Time: {end - start:.4f}s")
    return ret


def main():
    spark = SparkSession.builder.appName("PySpark Ray Equivalent").getOrCreate()

    run_experiment(spark, parallelism=100, size=10000, blowup=20)


if __name__ == "__main__":
    main()
