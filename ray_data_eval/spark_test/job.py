import sys

from humanize import naturalsize
import numpy as np
from pyspark.sql import SparkSession

NUM_PRODUCERS = 4
NUM_CONSUMERS = 0
NUM_EXECUTORS = NUM_PRODUCERS + NUM_CONSUMERS

spark = (
    SparkSession.builder.config("spark.driver.memory", "512m")
    .config("spark.executor.memory", "512m")
    .config("spark.executor.pyspark.memory", "4g")
    .config("spark.executor.instances", NUM_EXECUTORS)
    .config("spark.executor.cores", "1")
    .config("spark.dynamicAllocation.enabled", "false")
    .getOrCreate()
)

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

ROW_SIZE = 8  # int64
OUTPUT_SIZE = 1 * GB
OUTPUT_ROWS = OUTPUT_SIZE // ROW_SIZE
BLOCK_SIZE = 64 * MB
BLOCK_ROWS = BLOCK_SIZE // ROW_SIZE


def produce(part):
    part = list(part)
    part_id = part[0]
    start = part_id * OUTPUT_ROWS
    return [np.arange(start, start + OUTPUT_ROWS, dtype=np.uint64)]


def consume(part):
    part = list(part)
    print("Consume: partition size", naturalsize(sys.getsizeof(part)))
    return [len(part)]


def produceStreaming(part):
    # part = list(part)
    # part_id = part[0]
    part_id = part
    start = part_id * OUTPUT_ROWS
    num_rows = 0
    while num_rows < OUTPUT_ROWS:
        num_rows += BLOCK_ROWS
        print(part_id, "Yield", start, start + BLOCK_ROWS)
        yield np.arange(start, start + BLOCK_ROWS, dtype=np.uint64)
        start += BLOCK_ROWS


def consumeStreaming(part):
    count = 0
    for row in part:
        count += len(row)
        print("Consume", len(row))
    print("Consume returns", count)
    return [count]


rdd = spark.sparkContext.parallelize(range(NUM_PRODUCERS))
print(f"Input: {rdd.count():_} rows")
# rdd = (
#     rdd.repartition(NUM_PRODUCERS)
#     .mapPartitions(produce)
#     .flatMap(lambda x: x)
#     .mapPartitions(consume)
# )
rdd = (
    rdd.repartition(NUM_PRODUCERS).flatMap(produceStreaming).cache().mapPartitions(consumeStreaming)
)
print("Collected:", rdd.collect())
row_count = rdd.count()
print(f"Output: {row_count:_} rows")
spark.stop()
