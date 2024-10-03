# Ejecucion en cluster

El codigo para entrenar el modelo requiere de la creacion de un cluster de kind usando los siguientes comandos

```
kind create cluster --config .\work-station\config.yaml
```
Y luego se debe instalar el cluster con el siguiente comando

```
helm install my-release oci://registry-1.docker.io/bitnamicharts/spark --values .\Training_Comparison\helm.yaml
```
este comando usa un archivo `helm.yaml` que sobre escribe la imagen normal que contiene las librerias correspondientes `Dockerfile.spark`.

posteriormente debemos conectarnos a nodo master para correr el script o usar el run.sh para correrlo desde la linea de comandos

```
kubectl exec -n default my-release-spark-master-0 -c spark-master -- /bin/sh -c "/home/spark/tensorflow_cluster/run.sh tensorflow_cluster.py"
```


