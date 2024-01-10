NUM_TASK_MANAGERS=19

for ((i=0; i<$NUM_TASK_MANAGERS; i++)); do
    ../flink-1.18.0/bin/taskmanager.sh start
    sleep 0.1
done
