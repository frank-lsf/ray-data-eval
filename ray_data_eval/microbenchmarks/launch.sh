
echo "Flink"
cd flink
bash launch.sh

cd ..
echo "Spark" 
cd spark
bash launch.sh

cd ..
echo "Tfdata"
cd tfdata
bash launch.sh

cd ..
echo "Spark Streaming"
cd spark_streaming
bash launch.sh

cd ..
cd raydata
echo "Ray Data"
bash launch.sh
