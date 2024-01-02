"""
Testing Spark structured streaming execution
"""

import os
import time

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, BinaryType


DATA_SIZE_BYTES = 1000 * 1000 * 100  # 100 MB
TIME_BASIS = 0.1  # How many seconds should time_factor=1 take


def memory_blowup(row, time_factor=1):
    i = row["item"]
    data = bytearray(np.full(DATA_SIZE_BYTES, 1, dtype=np.uint8))
    time.sleep(TIME_BASIS * time_factor)
    return {"data": data, "idx": i}


def memory_shrink(row, time_factor=1):
    data = row["data"]
    np_data = np.frombuffer(data, dtype=np.uint8)
    time.sleep(TIME_BASIS * time_factor)
    return (int(np_data.sum()),)


def run_experiment(spark, parallelism=-1, num_parts=100, producer_time=1, consumer_time=1):
    start = time.perf_counter()

    items = [(item,) for item in range(num_parts)]
    input_schema = ["item"]
    df = spark.createDataFrame(data=items, schema=input_schema)
    df = df.repartition(parallelism)

    blowup_schema = StructType(
        [
            StructField("data", BinaryType(), True),
            StructField("idx", IntegerType(), True),
        ]
    )
    df = df.rdd.map(lambda row: memory_blowup(row, time_factor=producer_time)).toDF(blowup_schema)

    result_schema = StructType([StructField("result", IntegerType(), True)])
    df = df.rdd.map(lambda row: memory_shrink(row, time_factor=consumer_time)).toDF(result_schema)

    ret = df.agg({"result": "sum"}).collect()[0][0]

    run_time = time.perf_counter() - start
    print(f"\n{ret:,}")
    print(df.explain())
    print(run_time)


def main():
    spark = (
        SparkSession.builder.appName("Spark")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .config("spark.executor.memory", "20g")
        .config("spark.driver.memory", "20g")
        .config("spark.executor.instances", "100")
        .getOrCreate()
    )

    config = {
        "parallelism": 200,
        "total_data_size_gb": 100,
        "producer_time": 1,
        "consumer_time": 9,
    }
    config["total_data_size"] = config["total_data_size_gb"] * 10**9
    config["num_parts"] = config["total_data_size"] // DATA_SIZE_BYTES
    config["producer_consumer_ratio"] = config["producer_time"] / config["consumer_time"]

    run_experiment(
        spark,
        parallelism=config["parallelism"],
        num_parts=config["num_parts"],
        producer_time=config["producer_time"],
        consumer_time=config["consumer_time"],
    )


if __name__ == "__main__":
    main()
