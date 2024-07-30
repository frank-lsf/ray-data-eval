# High memory
python -u producer_consumer_gpu.py --mem-limit 10 > mem-limit-10.log 2>&1

# Low memory
python -u producer_consumer_gpu.py --mem-limit 6 > mem-limit-6.log 2>&1