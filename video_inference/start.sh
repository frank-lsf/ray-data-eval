#!/bin/bash

ray stop -f
pkill prometheus
pkill grafana-server

ray start --head --num-gpus=2
../bin/prometheus/prometheus --config.file=/tmp/ray/session_latest/metrics/prometheus/prometheus.yml &
sudo systemctl start grafana-server
