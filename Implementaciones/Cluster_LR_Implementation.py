from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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
        'accuracy' : accu
    }
    nombre_archivo = 'Cluster_tiempos.json'

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

# 2. Inicializar el modelo de regresión logística
lr = LogisticRegression(featuresCol="features", labelCol="label")

inicio = time.time()

model = lr.fit(df_train)

fin = time.time()
tiempo_ejecucion_LR = fin - inicio

# 5. Hacer predicciones
predictions = model.transform(df_test)

# 1. Evaluar precisión (accuracy)
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy:.2f}, tiempo: {tiempo_ejecucion_LR:.2f}")

writeJson(tiempo_ejecucion_LR,accuracy,"Logistic Regression")

# Cerrar la sesión de Spark
spark.stop()

