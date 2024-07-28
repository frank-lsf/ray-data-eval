import time
import os
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.listener import StreamingListener
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import StructType, StructField, IntegerType, BinaryType
from pyspark.sql.functions import udf

import argparse

NUM_CPUS = 8
NUM_GPUS = 4

MB = 1024 * 1024

NUM_ROWS_PER_TASK = 10
NUM_TASKS = 16 * 5
NUM_ROWS_TOTAL = NUM_ROWS_PER_TASK * NUM_TASKS
ROW_SIZE = 100 * MB

TIME_UNIT = 0.5



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
    executor_memory_in_mb = int(executor_memory * 1024 / (NUM_CPUS + NUM_GPUS))
    # https://spark.apache.org/docs/latest/configuration.html
    conf = (
        SparkConf()
        .setAppName("Local Spark Streaming Example")
        .setMaster(f"local[{NUM_CPUS}]")
        .set("spark.eventLog.enabled", "true")
        .set("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .set("spark.driver.memory", "2g")
        .set("spark.executor.memory", f"{executor_memory_in_mb}m")
        .set("spark.executor.instances", NUM_CPUS + NUM_GPUS)
        .set("spark.executor.cores", 1)
        .set("spark.cores.max", NUM_CPUS + NUM_GPUS)
        .set("spark.scheduler.mode", "FAIR")
    )
    BATCH_INTERVAL = 0.1  # seconds
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, BATCH_INTERVAL)
    return sc, ssc

def producer(row):
    time.sleep(TIME_UNIT * 10)
    for j in range(NUM_ROWS_PER_TASK):
        data = b"1" * ROW_SIZE
        yield (data, row[0] * NUM_ROWS_PER_TASK + j)


def consumer(batch_rows):
    time.sleep(TIME_UNIT)
    data = b"2" * ROW_SIZE
    return (data,)

def inference(row):
    time.sleep(TIME_UNIT)
    return 1


def run_spark_data(ssc, sql_context):

    # Define schema for incoming data
    schema = StructType([
        StructField("data", BinaryType(), True),
        StructField("id", IntegerType(), True)
    ])
    
    producer_udf = udf(producer, schema)
    
    consumer_udf = udf(consumer, BinaryType())

    inference_udf = udf(inference, IntegerType())
    
    rdd = ssc.sparkContext.parallelize([Row(id=i) for i in range(10)])
    dstream = ssc.queueStream([rdd])
    
    # Apply the UDFs in the streaming pipeline
    def process_stream(rdd):
        if not rdd.isEmpty():
            df = ssc.createDataFrame(rdd, schema=schema)
            
            # Apply Producer UDF
            produced_df = df.withColumn("produced_data", producer_udf(df["id"]))
            
            # Apply Consumer UDF
            consumed_df = produced_df.withColumn("consumed_data", consumer_udf(produced_df["produced_data"]))
            
            # Apply Inference UDF
            inferred_df = consumed_df.withColumn("inference_result", inference_udf(consumed_df["consumed_data"]))
            
            inferred_df.show()

    dstream.foreachRDD(lambda rdd: process_stream(rdd))

    # Start streaming context
    ssc.start()
    ssc.awaitTermination()

def _run_spark_data(ssc, sql_context):
    start = time.perf_counter()
    items = [(item,) for item in range(NUM_TASKS)]

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
