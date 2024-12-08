import json

data = []
with open('flink_logs_producer_consumer_gpu.log') as f:
    for line in f:
        line = json.loads(line)
        line['tid'] = line['cat']
        data.append(line)

json.dump(data, open('flink_logs_producer_consumer_gpu.json', 'w'), indent=2)