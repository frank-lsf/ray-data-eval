export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
export MICROBENCHMARK_HOME=/home/ubuntu/ray-data-eval/ray_data_eval/microbenchmarks

for mem_limit in {10..2..2}; do
    echo "Running $mem_limit GB"
    echo "Log: $MICROBENCHMARK_HOME/results/tfdata/mem-limit-${mem_limit}.log"
    python -u producer_consumer_gpu.py --mem-limit $mem_limit > $MICROBENCHMARK_HOME/results/tfdata/mem-limit-${mem_limit}.log 2>&1
done
