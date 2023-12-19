#!/bin/bash

# Latest stable release
FLINK_VERSION=1.18.0

# Create new conda env to install PyFlink
echo Activating conda environment "raydata"...
eval "$(conda shell.bash hook)"
conda activate raydata

if [ "$CONDA_DEFAULT_ENV" == "raydata" ]; then
    echo Conda environment "raydata" is activated.
else
    echo Failed to activate conda environment "raydata". Did you initialize?
fi

# -- PyFlink already comes with Flink runtime so we don't need to install Flink separately --
# FLINK_DOWNLOAD_URL="https://dlcdn.apache.org/flink/flink-1.18.0/flink-1.18.0-bin-scala_2.12.tgz"

# mkdir -p gen_flink
# cd gen_flink

# echo "Downloading Flink $FLINK_VERSION..."
# wget -c "$FLINK_DOWNLOAD_URL" -O "flink-$FLINK_VERSION.tgz"

# echo "Unzipping Flink..."
# tar -xzf "flink-$FLINK_VERSION.tgz" # -x extract; -z decompress with gzip; -f filename

echo Installing PyFlink...
pip install apache-flink==$FLINK_VERSION

echo Flink setup completed.
echo To start running flink, switch to the conda environment "raydata":
echo conda activate raydata
