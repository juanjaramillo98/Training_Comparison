#!/bin/bash

# source /tmp/venv/bin/activate 

spark-submit Cluster_LR_Implementation.py

spark-submit Cluster_MLPC_hiperparametros.py

spark-submit Cluster_NB_Implementation.py

spark-submit Cluster_LR_Implementation.py

spark-submit Cluster_MLPC_hiperparametros.py

spark-submit Cluster_NB_Implementation.py