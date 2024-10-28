# Ejecucion en cluster

## kindcluster

El codigo para entrenar el modelo requiere de la creacion de un cluster de kind usando los siguientes comandos
```
kind create cluster --config .\Configuration\config.yaml
```


## Custom Spark Image

Es necesario crear una imagen customizada con tensorflow instalado para poder entrenar modelos

```
docker build -f DockerFiles\Dockerfile.Spark -t custom_tensorflow_spark .
```

## helm chart

Y luego se debe instalar el cluster con el siguiente comando

```
helm install my-release oci://registry-1.docker.io/bitnamicharts/spark --values .\Configuration\helm.yaml --set securityContext.runAsUser=0
```
este comando usa un archivo `helm.yaml` que sobre escribe la imagen normal que contiene las librerias correspondientes `Dockerfile.spark`.

## Ejecucion del script

posteriormente debemos conectarnos al nodo master para correr el script 

```
kubectl exec -i -t -n default my-release-spark-master-0 -c spark-master -- sh -c "clear; (bash || ash || sh)"
```
estando en la consola del nodo master se ejecuta el train 

```
sh /home/spark/run.sh train_model.py
```
# Ejecucion en CPU o GPU

Para la ejecucion del python en un ambiente normal se usa una imagen custom que parte desde la imagen `tensorflow/tensorflow:latest-gpu-jupyter` usando `Dockerfile.GPU`

```
docker build -f DockerFiles\Dockerfile.GPU -t gpu_execute .
```

## CPU
```
docker run --rm -v C:\Users\nitro\OneDrive\Escritorio\Freestyle\Spark\Training_Comparison:/home/spark gpu_execute:latest
```

## GPU

```
docker run --rm --runtime=nvidia -v C:\Users\nitro\OneDrive\Escritorio\Freestyle\Spark\Training_Comparison:/home/spark gpu_execute:latest
```



