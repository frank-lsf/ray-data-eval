set -ex

export SPARK_EVENTS_PATH=/home/ubuntu/ray-data-eval/logs/spark-events
export SPARK_TRACE_EVENT_PATH=/home/ubuntu/ray-data-eval/logs/spark-trace-events
export SPARK_EVENTS_FILEURL=file://$SPARK_EVENTS_PATH
export SPARK_HOME=/home/ubuntu/miniconda3/envs/raydata/lib/python3.10/site-packages/pyspark
export PYSPARK_PYTHON=/home/ubuntu/miniconda3/envs/raydata/bin/python
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=$SPARK_EVENTS_FILEURL"

$SPARK_HOME/sbin/stop-history-server.sh
$SPARK_HOME/sbin/start-history-server.sh
python producer_consumer_microbenchmark.py
