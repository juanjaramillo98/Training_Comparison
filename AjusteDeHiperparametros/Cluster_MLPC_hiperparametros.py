from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

import numpy as np
import pathlib
import time
import os
import json

# Iniciar una sesión de Spark
spark = SparkSession.builder.appName("LogisticRegressionExample").getOrCreate()


#----------------------------------------------------------Data ingestion-----------------------------------------------

path1 = 'DatasetVGG19/test.npy'
path2 = 'DatasetVGG19/train.npy'
path3 = 'DatasetVGG19/validation.npy'
localPath = pathlib.Path().resolve()

train_data_path = pathlib.Path(localPath ,path2)
test_data_path = pathlib.Path(localPath ,path1)

print(train_data_path)

# Cargar el dataset en formato .npy y convertirlo a un DataFrame de PySpark

def unpack(array):
    x = array[:,:-2]
    y = array[:,-2:]
    y = np.argmax(y, axis=1)
    rows = [Row(features=Vectors.dense(f), label=int(l)) for f, l in zip(x, y)]
    df = spark.createDataFrame(rows)
    return df

df_train = unpack(np.load(train_data_path))
df_test = unpack(np.load(test_data_path))

def writeJson(tiempo,accu,metod):
    nuevo_registro = {
        'Tipo Ejecucion' : "Cluster",
        'Metodo' : metod,
        'Epochs':20,
        'tiempo_ejecucion': tiempo,
        'accuracy' : accu,
        'parallelism' : 2
    }
    nombre_archivo = 'Clus_MLPC_Tunning.json'

    # Leer el contenido existente, si el archivo ya existe
    if os.path.exists(nombre_archivo):
        with open(nombre_archivo, 'r') as archivo:
            try:
                registros = json.load(archivo)
            except json.JSONDecodeError:
                registros = []
    else:
        registros = []

    # Agregar el nuevo registro
    registros.append(nuevo_registro)

    # Guardar los registros actualizados en el archivo JSON
    with open(nombre_archivo, 'w') as archivo:
        json.dump(registros, archivo, indent=4)

#-----------------------------------------------------------LogisticRegression----------------------------------------------------------

mlpc = MultilayerPerceptronClassifier()
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

ParamMap = (
    ParamGridBuilder()
    .addGrid(
        mlpc.layers, [
            [1000, 500, 2],
            [1000, 100, 2],
            [1000, 500,  200, 2],
        ]
    )
    .addGrid(mlpc.stepSize, [0.001, 0.0001])
    .baseOn({mlpc.maxIter : 20})
    .baseOn({mlpc.solver : "gd"})
    .baseOn({mlpc.featuresCol : "features"})
    .baseOn({mlpc.labelCol : "label"})
    .build()
)

cv = CrossValidator(
    estimator = mlpc, 
    estimatorParamMaps = ParamMap, 
    evaluator = accuracy_evaluator,
    parallelism = 2
)


inicio = time.time()

# 3. Entrenar el modelo
cvModel = cv.fit(df_train)

fin = time.time()
tiempo_ejecucion_MLPC = fin - inicio

print(cvModel.avgMetrics)

print(ParamMap)

print(f"Accuracy: {cvModel.avgMetrics[0]:.2f}, tiempo: {tiempo_ejecucion_MLPC:.2f}")

writeJson(tiempo_ejecucion_MLPC,cvModel.avgMetrics[0],"MLPC")

spark.stop()
