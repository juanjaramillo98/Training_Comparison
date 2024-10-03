#!/bin/bash

# source /tmp/venv/bin/activate 

cd

cd /home/spark/tensorflow_cluster

ls

/opt/bitnami/spark/bin/spark-submit     \
  --class org.apache.spark.examples.SparkPi     \
  --conf spark.kubernetes.container.image=bitnami/spark:3     \
  --master spark://my-release-spark-master-svc.default.svc.cluster.local:7077    \
  --conf spark.kubernetes.driverEnv.SPARK_MASTER_URL=spark://my-release-spark-master-svc.default.svc.cluster.local:7077       \
  $1