#!/bin/bash

# Download from s3 bucket into a folder ~/imagenet

cd ~/imagenet/ILSVRC/Data/val

# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash