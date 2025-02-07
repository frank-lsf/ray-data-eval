MICROBENCHMARK_HOME=/home/ubuntu/ray-data-eval/ray_data_eval/microbenchmarks

for mem_limit in 16 14 12 10 8 6; do
    for systems in raydata; do

        log_file="$MICROBENCHMARK_HOME/results_ray_no_partition/$systems-mem-limit-${mem_limit}"

        # Check if the "Memory exceeded!" message is NOT in the log file
        if [ -f "$log_file" ]; then
            # if ! grep -q "Memory exceeded!" "$log_file"; then
                # echo "Memory did not exceed for $mem_limit GB, continuing to the next iteration."
            if grep -q "Run time:" $log_file; then
                echo "Skip: $log_file"
                continue  # Continue if memory was not exceeded
            fi
            # fi
        else
            echo "Log file $log_file does not exist."
        fi
        cd /home/ubuntu/ray-data-eval/ray_data_eval/microbenchmarks
        echo $systems
        cd $systems
        bash launch_no_partition.sh $mem_limit $log_file
    done
done