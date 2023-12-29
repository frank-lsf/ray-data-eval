package main.spark;

import java.nio.file.Paths;
import java.io.IOException;
import java.nio.file.Files;
import java.util.stream.Stream;
import java.util.stream.Collectors;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;

public class LoadImagesSpark {
    private static final String DATA_INDEX = "/mnt/data/ray-data-eval/index/train.txt";
    private static final String DATA_ROOT = "/mnt/data/ray-data-eval";
    private static final int TOTAL_COUNT = 1281167;

    private static String[] loadFilenames(int limit) {
        try (Stream<String> stream = Files.lines(Paths.get(DATA_INDEX))) {
            List<String> paths = stream
                    .map(path -> DATA_ROOT + "/" + path)
                    .limit(limit > 0 ? limit : TOTAL_COUNT)
                    .collect(Collectors.toList());
            return paths.toArray(new String[0]);
        } catch (IOException e) {
            e.printStackTrace();
            return new String[0];
        }
    }

    public static void testLoad(SparkSession spark, int limit) {
        String[] filenames = loadFilenames(limit);

        System.out.println("Start");
        long start = System.currentTimeMillis();

        Dataset<Row> df = spark.read().format("binaryFile").load(filenames).select("content");

        Dataset<Row> sizeDf = df.withColumn("size", length(col("content")));
        long totalSize = sizeDf.groupBy().sum("size").collectAsList().get(0).getLong(0);

        long end = System.currentTimeMillis();
        System.out.println("\n" + totalSize);
        System.out.println("Time: " + (end - start) / 1000.0 + "s");
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("load-images-spark")
                .config("spark.driver.memory", "8g")
                .config("spark.default.parallelism", "100")
                .config("spark.master", "local")
                .getOrCreate();

        testLoad(spark, 10000);
        spark.stop();
    }
}
