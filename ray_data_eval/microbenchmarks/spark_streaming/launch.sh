set -ex

export SPARK_EVENTS_PATH=/home/ubuntu/ray-data-eval/logs/spark-events
export SPARK_TRACE_EVENT_PATH=/home/ubuntu/ray-data-eval/logs/spark-trace-events
export SPARK_EVENTS_FILEURL=file://$SPARK_EVENTS_PATH
export SPARK_HOME=/home/ubuntu/ray-data-eval/ray_data_eval/microbenchmarks/spark/spark-3.5.1-bin-hadoop3
export PYSPARK_PYTHON=/opt/conda/envs/ray-data/bin/python
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=$SPARK_EVENTS_FILEURL"
export MICROBENCHMARK_HOME=/home/ubuntu/ray-data-eval/ray_data_eval/microbenchmarks

export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=4 -Dspark.worker.resource.gpu.discoveryScript=./gpu_discovery.sh -Dspark.executor.instances=8 -Dspark.executor.cores=1"
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

$SPARK_HOME/sbin/stop-history-server.sh
$SPARK_HOME/sbin/stop-master.sh
$SPARK_HOME/sbin/stop-worker.sh 

$SPARK_HOME/sbin/start-history-server.sh
$SPARK_HOME/sbin/start-master.sh --host localhost
$SPARK_HOME/sbin/start-worker.sh spark://localhost:7077

for mem_limit in {20..4..4}; do
    echo "Running $mem_limit GB"

    log_file="$MICROBENCHMARK_HOME/results/spark-streaming-mem-limit-${mem_limit}"

    # Check if the "Memory exceeded!" message is NOT in the log file
    if [ -f "$log_file" ]; then
        if ! grep -q "Memory exceeded!" "$log_file"; then
            echo "Memory did not exceed for $mem_limit GB, continuing to the next iteration."
            if grep -q "runtime" "$log_file"; then
                continue  # Continue if memory was not exceeded
            fi
        fi
    else
        echo "Log file $log_file does not exist."
    fi

    echo "Log: $log_file"

    python -u producer_consumer_gpu.py --mem-limit $mem_limit --stage-level-scheduling > $log_file 2>&1

    # Check if the "Memory exceeded!" message is NOT in the log file
    if [ -f "$log_file" ]; then
        if grep -q "Memory exceeded!" "$log_file"; then
            break
        fi
    fi
done

