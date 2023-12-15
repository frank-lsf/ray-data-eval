#!/bin/bash

# Latest stable release 
SPARK_VERSION="3.5.0"

# Download URL
SPARK_URL="https://dlcdn.apache.org/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop3.tgz"

# Installation directory
INSTALL_DIR="bin/"

# Download Spark
echo "Downloading Spark..."
wget $SPARK_URL

# Create installation directory if not exists
mkdir -p $INSTALL_DIR

# Unzip Spark to the installation directory
echo "Unzipping Spark..."
tar -xzf spark-$SPARK_VERSION-bin-hadoop3.tgz -C $INSTALL_DIR

# Rename Spark directory to 'spark'
mv $INSTALL_DIR/spark-$SPARK_VERSION-bin-hadoop3 $INSTALL_DIR/spark

# Clean up downloaded archive
rm spark-$SPARK_VERSION-bin-hadoop3.tgz

echo "Spark $SPARK_VERSION has been installed and configured in $INSTALL_DIR/spark"