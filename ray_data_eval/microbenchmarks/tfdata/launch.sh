export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

mem_limit=$1
log_file=$2

echo $log_file
echo "Running $mem_limit GB"
python -u producer_consumer_gpu.py --mem-limit $mem_limit > $log_file 2>&1
