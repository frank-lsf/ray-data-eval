import time
import os
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.listener import StreamingListener

import argparse
import sys
from setting import (
    TIME_UNIT,
    FRAMES_PER_VIDEO,
    NUM_VIDEOS,
    FRAME_SIZE_B,
)

parent_directory = os.path.abspath("..")
sys.path.append(parent_directory)


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
    # https://spark.apache.org/docs/latest/configuration.html
    conf = (
        SparkConf()
        .set("spark.dynamicAllocation.enabled", "false")
        .set("spark.executor.instances", 4)
        .set("spark.executor.cores", 2)
        .set("spark.executor.memory", "5g")
        .set("spark.driver.memory", "8g")  # Increase driver memory if necessary
        # Allocate 1 GPU per executor.
        .set("spark.executor.resource.gpu.amount", 1)
    )
    BATCH_INTERVAL = 0.1  # seconds
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, BATCH_INTERVAL)
    return sc, ssc


def producer(row):
    time.sleep(TIME_UNIT * 10)
    for j in range(FRAMES_PER_VIDEO):
        data = b"1" * FRAME_SIZE_B
        yield (data, row * FRAMES_PER_VIDEO + j)


def consumer(row):
    time.sleep(TIME_UNIT)
    data = b"2" * FRAME_SIZE_B
    return (data,)


def inference(row):
    time.sleep(TIME_UNIT)
    return 1


def run_spark_data(ssc, mem_limit):
    start = time.perf_counter()
    BATCH_SIZE = NUM_VIDEOS if mem_limit >= 10 else 1
    rdd_queue = [
        ssc.sparkContext.range(i, i + BATCH_SIZE) for i in range(0, NUM_VIDEOS, BATCH_SIZE)
    ]
    input_stream = ssc.queueStream(rdd_queue)

    # Apply the producer UDF
    producer_stream = input_stream.flatMap(producer)

    def process_batch(rdd):
        if rdd.isEmpty():
            run_time = time.perf_counter() - start
            print(f"\nTotal runtime of streaming computation: {run_time:.2f} seconds")
            ssc.stop(stopSparkContext=True, stopGraceFully=False)
            return

        consumer_rdd = rdd.map(lambda x: consumer(x)).cache()
        inference_rdd = consumer_rdd.map(lambda x: inference(x)).cache()

        total_processed = inference_rdd.count()
        print(f"Total length of data processed in batch: {total_processed:,}")

    producer_stream.foreachRDD(process_batch)


def bench(mem_limit):
    sc, ssc = start_spark_streaming(mem_limit)
    listener = StreamingListener()
    ssc.addStreamingListener(listener)

    run_spark_data(ssc, mem_limit)
    ssc.start()
    ssc.awaitTerminationOrTimeout(500)
    print("Stopping streaming context.")
    ssc.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=30
    )
    args = parser.parse_args()

    bench(args.mem_limit)
