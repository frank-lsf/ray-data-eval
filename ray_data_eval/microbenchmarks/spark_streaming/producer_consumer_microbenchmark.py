import time
import os
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.listener import StreamingListener
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import StructType, StructField, IntegerType

NUM_CPUS = 8
MB = 1024 * 1024

NUM_TASKS = 16 * 5
TIME_UNIT = 0.5

BLOCK_SIZE = 1 * MB
NUM_ROWS_PER_PRODUCER = 1000
NUM_ROWS_PER_CONSUMER = 100


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


def start_spark():
    conf = (
        SparkConf()
        .setAppName("Local Spark Streaming Example")
        .setMaster(f"local[{NUM_CPUS}]")
        .set("spark.eventLog.enabled", "true")
        .set("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .set("spark.driver.memory", "2g")
        .set("spark.executor.memory", "2g")
        .set("spark.cores.max", NUM_CPUS)
        .set("spark.scheduler.mode", "FAIR")
    )
    BATCH_INTERVAL = 0.1  # seconds
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, BATCH_INTERVAL)
    return sc, ssc


def busy_loop_for_seconds(time_diff):
    start = time.perf_counter()
    i = 0
    while time.perf_counter() - start < time_diff:
        i += 1
        continue


def producer_udf(row):
    # print("producer", row)
    # Simulate a delay
    busy_loop_for_seconds(TIME_UNIT * 10)
    for j in range(NUM_ROWS_PER_PRODUCER):
        data = b"1" * BLOCK_SIZE
        yield (data, row[0] * NUM_ROWS_PER_PRODUCER + j)


def consumer_udf(batch_rows):
    # print("consumer")
    # Simulate a delay
    busy_loop_for_seconds(TIME_UNIT * 1 / NUM_ROWS_PER_CONSUMER)
    return (int(len(batch_rows)),)


def run_spark_data(ssc, sql_context):
    start = time.perf_counter()
    items = [(item,) for item in range(NUM_TASKS)]

    # Create a DStream from a queue of RDDs
    rdd_queue = [ssc.sparkContext.parallelize(items, NUM_CPUS)]
    input_stream = ssc.queueStream(rdd_queue, oneAtATime=True)

    # Apply the producer UDF
    producer_stream = input_stream.flatMap(producer_udf)

    def process_batch(rdd):
        if rdd.isEmpty():
            run_time = time.perf_counter() - start
            print(f"\nTotal runtime of streaming computation: {run_time:.2f} seconds")
            ssc.stop(stopSparkContext=False, stopGraceFully=True)
            return
        df = sql_context.createDataFrame(rdd, ["data", "id"])
        consumer_rdd = df.rdd.map(lambda x: consumer_udf(x))
        result_schema = StructType([StructField("result", IntegerType(), True)])
        consumer_df = consumer_rdd.map(lambda x: Row(result=x[0])).toDF(result_schema)
        total_processed = consumer_df.agg({"result": "sum"}).collect()[0][0]
        print(f"\nTotal length of data processed in batch: {total_processed:,}")

    producer_stream.foreachRDD(process_batch)


def run_experiment():
    sc, ssc = start_spark()
    sql_context = SQLContext(sc)
    listener = CustomStreamingListener()
    ssc.addStreamingListener(listener)

    run_spark_data(ssc, sql_context)
    ssc.start()
    ssc.awaitTerminationOrTimeout(3600)


def main():
    run_experiment()


if __name__ == "__main__":
    main()
