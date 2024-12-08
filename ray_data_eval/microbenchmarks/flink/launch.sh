export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

mem_limit=$1
echo "Running $mem_limit GB"
log_file=$2

echo "Log: $log_file"
python -u producer_consumer_gpu.py --mem-limit $mem_limit > $log_file 2>&1

