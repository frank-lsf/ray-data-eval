#!/bin/bash

nohup python main.py -a resnet50 -b 128 ~/imagenet/ILSVRC/Data/CLS-LOC > output.log 2>&1 &