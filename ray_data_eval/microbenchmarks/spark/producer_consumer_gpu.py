import time

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, BinaryType
import os
import argparse

NUM_CPUS = 8
NUM_GPUS = 4

MB = 1024 * 1024

NUM_ROWS_PER_TASK = 10
NUM_TASKS = 16 * 5
NUM_ROWS_TOTAL = NUM_ROWS_PER_TASK * NUM_TASKS
ROW_SIZE = 100 * MB

TIME_UNIT = 0.5


def start_spark(executor_memory: int):
    executor_memory_in_mb = int(executor_memory * 1024 / (NUM_CPUS + NUM_GPUS))
    # https://spark.apache.org/docs/latest/configuration.html
    spark = (
        SparkSession.builder.appName("Local Spark Example")
        .master(f"local[{NUM_CPUS + NUM_GPUS}]")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .config("spark.driver.memory", "2g")
        # Amount of memory to use per executor process
        .config("spark.executor.memory", f"{executor_memory_in_mb}m")
        .config("spark.executor.instances", NUM_CPUS + NUM_GPUS)
        # The number of cores to use on each executor. 
        .config("spark.executor.cores", 1)
        # The maximum amount of CPU cores to request for the application from across the cluster (not from each machine).
        .config("spark.cores.max", NUM_CPUS + NUM_GPUS)
        .getOrCreate()
    )
    return spark


def producer_udf(row):
    time.sleep(TIME_UNIT * 10)
    for j in range(NUM_ROWS_PER_TASK):
        data = b"1" * ROW_SIZE
        yield (data, row.item * NUM_ROWS_PER_TASK + j)


def consumer_udf(row):
    time.sleep(TIME_UNIT)
    data = b"2" * ROW_SIZE
    return (data,)


def inference_udf(row):
    time.sleep(TIME_UNIT)
    return 1


def run_spark_data(spark):
    start = time.perf_counter()

    items = [(item,) for item in range(NUM_TASKS)]
    input_schema = StructType([StructField("item", IntegerType(), True)])
    df = spark.createDataFrame(items, schema=input_schema)

    print("df.count()", df.count())

    producer_schema = StructType(
        [StructField("data", BinaryType(), True), StructField("id", IntegerType(), True)]
    )
    producer_rdd = df.rdd.flatMap(producer_udf)
    producer_df = spark.createDataFrame(producer_rdd, schema=producer_schema)

    print("producer_df.count()", producer_df.count())

    consumer_schema = StructType([StructField("data", BinaryType(), True)])
    consumer_rdd = producer_df.rdd.map(consumer_udf)
    consumer_df = spark.createDataFrame(consumer_rdd, schema=consumer_schema)

    print("consumer_df.count()", consumer_df.count())

    result_schema = StructType([StructField("result", IntegerType(), True)])
    inference_rdd = consumer_df.rdd.map(lambda row: (inference_udf(row),))
    inference_df = spark.createDataFrame(inference_rdd, schema=result_schema)

    print("inference_df.count()", inference_df.count())

    total_processed = inference_df.agg({"result": "sum"}).collect()[0][0]

    run_time = time.perf_counter() - start
    print(f"\nTotal length of data processed: {total_processed:,}")
    print(f"Run time: {run_time:.2f} seconds")
    return total_processed


def bench(mem_limit):
    spark = start_spark(mem_limit)
    run_spark_data(spark)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=30
    )
    args = parser.parse_args()

    bench(args.mem_limit)
