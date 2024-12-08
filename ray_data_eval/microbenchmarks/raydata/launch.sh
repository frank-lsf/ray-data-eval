export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
export MICROBENCHMARK_HOME=/home/ubuntu/ray-data-eval/ray_data_eval/microbenchmarks

for mem_limit in {12..4..4}; do
    echo "Running $mem_limit GB"


    log_file="$MICROBENCHMARK_HOME/results/raydata-mem-limit-${mem_limit}"

    # Check if the "Memory exceeded!" message is NOT in the log file
    if [ -f "$log_file" ]; then
        if ! grep -q "Memory exceeded!" "$log_file"; then
            echo "Memory did not exceed for $mem_limit GB, continuing to the next iteration."
            if grep -q "Run time: " "$log_file"; then
                continue  # Continue if memory was not exceeded
            fi
        fi
    else
        echo "Log file $log_file does not exist."
    fi

    echo "Log: $log_file"


    python -u producer_consumer_gpu.py --mem-limit $mem_limit > $log_file 2>&1

    # Check if the "Memory exceeded!" message is NOT in the log file
    if [ -f "$log_file" ]; then
        if grep -q "Memory exceeded!" "$log_file"; then
            break
        fi
    fi
done
