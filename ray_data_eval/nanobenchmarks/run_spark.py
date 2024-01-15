import time
import os

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType

from ray_data_eval.common.types import test_problem

MB = 1000 * 1000
DATA_SIZE_BYTES = 100 * MB  # 100 MB
TIME_UNIT = 1  # seconds


def start_spark(cfg):
    memory_limit_mb = (cfg.buffer_size_limit * DATA_SIZE_BYTES * 9) // MB
    memory_limit = str(memory_limit_mb) + "m"

    spark = (
        SparkSession.builder.appName("Spark")
        .config("spark.master", "spark://ec2-35-85-195-144.us-west-2.compute.amazonaws.com:7077")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .config("spark.executor.memory", "3g")
        .config("spark.driver.memory", "3g")
        .config("spark.cores.max", cfg.num_execution_slots)
        .getOrCreate()
    )
    return spark


def producer_udf(row, cfg):
    i = row["item"]
    data = b"1" * (DATA_SIZE_BYTES * cfg.producer_output_size[i])

    time.sleep(TIME_UNIT * cfg.producer_time[i])
    return {"data": data, "idx": i}


def consumer_udf(row, cfg):
    data = row["data"]
    time.sleep(TIME_UNIT * cfg.consumer_time[row["idx"]])
    return (int(len(data)),)


def run_spark_data(spark, cfg):
    if cfg.num_producers != cfg.num_consumers:
        raise NotImplementedError(f"num_producers != num_consumers: {cfg}")
    start = time.perf_counter()

    items = [(item,) for item in range(cfg.num_producers)]
    input_schema = ["item"]
    df = spark.sparkContext.parallelize(items, cfg.num_producers).toDF(input_schema)
    df = df.rdd.map(lambda row: producer_udf(row, cfg)).toDF()

    df.count()

    result_schema = StructType([StructField("result", IntegerType(), True)])
    df = df.rdd.map(lambda row: consumer_udf(row, cfg)).toDF(result_schema)

    ret = df.agg({"result": "sum"}).collect()[0][0]

    run_time = time.perf_counter() - start
    print(f"\n{ret:,}")
    print(df.explain())
    print(run_time)
    return ret


def run_experiment(cfg):
    spark = start_spark(cfg)
    run_spark_data(spark, cfg)


def main():
    run_experiment(test_problem)


if __name__ == "__main__":
    main()
