import time
import os
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.listener import StreamingListener
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import StructType, StructField, IntegerType

import argparse

import os
import sys
parent_directory = os.path.abspath('..')
sys.path.append(parent_directory)
from setting import *


class CustomStreamingListener(StreamingListener):
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def onBatchStarted(self, batchStarted):
        if self.start_time is None:
            self.start_time = batchStarted.batchInfo().submissionTime() / 1000.0
            print(f"Start time: {self.start_time:.2f}")

    def onBatchCompleted(self, batchCompleted):
        self.end_time = batchCompleted.batchInfo().submissionTime() / 1000.0
        print(
            f"\nTotal runtime of streaming computation: {self.end_time - self.start_time:.2f} seconds"
        )


def start_spark_streaming(executor_memory):
    NUM_EXECUTORS = 4
    executor_memory_in_mb = int(executor_memory * 1024 / NUM_EXECUTORS)
    # https://spark.apache.org/docs/latest/configuration.html
    conf = (
        SparkConf()
        .setAppName("Local Spark Streaming Example")
        .setMaster(f"local[{NUM_CPUS}]")
        .set("spark.eventLog.enabled", "true")
        .set("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .set("spark.driver.memory", "2g")
        .set("spark.executor.memory", f"{executor_memory_in_mb}m")
        .set("spark.executor.instances", NUM_EXECUTORS)  # 4 executors
        .set("spark.executor.cores", NUM_CPUS / NUM_EXECUTORS)      # Number of cores per executor
        .set("spark.dynamicAllocation.enabled", "false")  # Disable dynamic allocation
        .set("spark.executor.resource.gpu.amount", 1)     # Allocate 1 GPU per executor
        .set("spark.scheduler.mode", "FAIR")
    )
    BATCH_INTERVAL = 0.1  # seconds
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, BATCH_INTERVAL)
    return sc, ssc


def producer(row):
    time.sleep(TIME_UNIT * 10)
    for j in range(FRAMES_PER_VIDEO):
        data = b"1" * FRAME_SIZE_B
        yield (data, row[0] * FRAMES_PER_VIDEO + j)


def consumer(batch_rows):
    time.sleep(TIME_UNIT)
    data = b"2" * FRAME_SIZE_B
    return (data,)


def inference(row):
    time.sleep(TIME_UNIT)
    return 1


def run_spark_data(ssc, sql_context):
    start = time.perf_counter()
    items = [(item,) for item in range(NUM_VIDEOS)]

    # Create a DStream from a queue of RDDs
    rdd_queue = [ssc.sparkContext.parallelize(items, NUM_CPUS)]
    input_stream = ssc.queueStream(rdd_queue, oneAtATime=True)

    # Apply the producer UDF
    producer_stream = input_stream.flatMap(producer)

    def process_batch(rdd):
        if rdd.isEmpty():
            run_time = time.perf_counter() - start
            print(f"\nTotal runtime of streaming computation: {run_time:.2f} seconds")
            ssc.stop(stopSparkContext=False, stopGraceFully=True)
            return
        df = sql_context.createDataFrame(rdd, ["data", "id"])
        consumer_rdd = df.rdd.map(lambda x: consumer(x))
        inference_rdd = consumer_rdd.map(lambda x: inference(x))
        result_schema = StructType([StructField("result", IntegerType(), True)])
        inference_df = inference_rdd.map(lambda x: Row(result=x[0])).toDF(result_schema)
        total_processed = inference_df.agg({"result": "sum"}).collect()[0][0]
        print(f"\nTotal length of data processed in batch: {total_processed:,}")

    producer_stream.foreachRDD(process_batch)


def bench(mem_limit):
    sc, ssc = start_spark_streaming(mem_limit)
    sql_context = SQLContext(sc)
    listener = CustomStreamingListener()
    ssc.addStreamingListener(listener)

    run_spark_data(ssc, sql_context)
    ssc.start()
    ssc.awaitTerminationOrTimeout(3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=30
    )
    args = parser.parse_args()

    bench(args.mem_limit)
