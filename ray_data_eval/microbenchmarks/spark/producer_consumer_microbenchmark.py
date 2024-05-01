import time
import os

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, BinaryType

NUM_CPUS = 8
MB = 1024 * 1024

NUM_ROWS_PER_TASK = 10
NUM_TASKS = 16 * 5
NUM_ROWS_TOTAL = NUM_ROWS_PER_TASK * NUM_TASKS
BLOCK_SIZE = 100 * MB
TIME_UNIT = 0.1


def start_spark():
    spark = SparkSession.builder \
        .appName("Local Spark Example") \
        .master(f"local[{NUM_CPUS}]") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "12g") \
        .getOrCreate()
    return spark


def producer_udf(row):
    i = row.item
    # Simulate a delay
    time.sleep(TIME_UNIT * 10)
    # Yield blocks of data one by one
    for j in range(10):  # Yield 10 blocks of data
        data = b"1" * BLOCK_SIZE
        yield (data, i, j)

def consumer_udf(row):
    data, idx, jdx = row
    # Simulate processing time
    time.sleep(TIME_UNIT * 1)
    return (len(data), idx, jdx)

def run_spark_data(spark):
    start = time.perf_counter()

    # Creating initial DataFrame
    items = [(item,) for item in range(NUM_TASKS)]
    input_schema = StructType([StructField("item", IntegerType(), True)])
    df = spark.createDataFrame(items, schema=input_schema)

    # Applying the producer UDF with flatMap to handle multiple outputs per input
    producer_rdd = df.rdd.flatMap(producer_udf)
    producer_schema = StructType([
        StructField("data", BinaryType(), True),
        StructField("idx", IntegerType(), True),
        StructField("jdx", IntegerType(), True)
    ])
    producer_df = spark.createDataFrame(producer_rdd, schema=producer_schema)

    # Applying the consumer UDF
    consumer_rdd = producer_df.rdd.map(consumer_udf)
    result_schema = StructType([
        StructField("result", IntegerType(), True),
        StructField("idx", IntegerType(), True),
        StructField("jdx", IntegerType(), True)
    ])
    consumer_df = spark.createDataFrame(consumer_rdd, schema=result_schema)

    # Aggregate and collect the results
    ret = consumer_df.agg({"result": "sum"}).collect()[0][0]

    run_time = time.perf_counter() - start
    print(f"\nTotal length of data processed: {ret:,}")
    consumer_df.explain()
    print(f"Run time: {run_time:.2f} seconds")
    return ret


def run_experiment():
    spark = start_spark()
    run_spark_data(spark)


def main():
    run_experiment()


if __name__ == "__main__":
    main()
