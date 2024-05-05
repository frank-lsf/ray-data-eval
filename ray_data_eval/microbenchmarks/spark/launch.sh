# export SPARK_EVENTS_PATH=/home/ubuntu/ray-data-eval/logs/spark-events
# export SPARK_TRACE_EVENT_PATH=/home/ubuntu/ray-data-eval/logs/spark-trace-events
export SPARK_EVENTS_FILEURL=/home/ubuntu/ray-data-eval/logs/spark-events
# export SPARK_HOME=/home/ubuntu/ray-data-eval/bin/spark
# export PYSPARK_PYTHON=/home/ubuntu/miniconda3/envs/raydata/bin/python
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=$SPARK_EVENTS_FILEURL"

python producer_consumer_microbenchmark.py