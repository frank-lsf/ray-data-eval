spark-submit \
  --master local[*] \
  --driver-memory 512m \
  --executor-memory 1g \
  --num-executors 4 \
  --executor-cores 1 \
  --conf spark.dynamicAllocation.enabled=false \
  job.py
