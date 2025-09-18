export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
export MICROBENCHMARK_HOME=/home/ubuntu/ray-data-eval/ray_data_eval/microbenchmarks

mem_limit=$1
log_file=$2

echo "Running $mem_limit GB"
echo "Log: $log_file"

python -u ray_data_no_partition.py --mem-limit $mem_limit > $log_file 2>&1
