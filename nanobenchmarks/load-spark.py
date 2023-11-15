import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import length

from data_utils import load_index

BENCHMARK_NAME = "load-images-spark"


def test_load(spark, *, limit: int = -1):
    filenames = load_index(limit=limit, prefix="file://")

    start = time.perf_counter()

    df = spark.read.format("binaryFile").load(filenames).select("content")
    size_df = df.withColumn("size", length(df["content"]))
    ret = size_df.groupBy().sum("size").collect()[0][0]

    end = time.perf_counter()
    print()
    print(ret)
    print(f"Time: {end - start:.4f}s")
    return ret


def main():
    spark = (
        SparkSession.builder.appName(BENCHMARK_NAME)
        .config("spark.driver.memory", "8g")
        .config("spark.default.parallelism", "100")
        .getOrCreate()
    )

    test_load(spark, limit=10000)


if __name__ == "__main__":
    main()
