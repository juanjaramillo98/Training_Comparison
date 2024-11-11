from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier  # Cambiado a RandomForestClassifier
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import numpy as np
import pathlib
import time
import os
import json

# Iniciar una sesión de Spark
spark = SparkSession.builder.appName("RandomForestExample").getOrCreate()

#----------------------------------------------------------Data ingestion-----------------------------------------------

path1 = 'DatasetVGG19/test.npy'
path2 = 'DatasetVGG19/train.npy'
path3 = 'DatasetVGG19/validation.npy'
localPath = pathlib.Path().resolve()

train_data_path = pathlib.Path(localPath, path2)
test_data_path = pathlib.Path(localPath, path1)

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

def writeJson(tiempo, accu, metod):
    nuevo_registro = {
        'Tipo Ejecucion' : "Cluster",
        'Metodo' : metod,
        'depth' : 10,
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

#-----------------------------------------------------------RandomForest----------------------------------------------------------

# 2. Inicializar el modelo de Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=10)  # Ajustar parámetros si es necesario

inicio = time.time()

model = rf.fit(df_train)

fin = time.time()
tiempo_ejecucion_RF = fin - inicio

# 5. Hacer predicciones
predictions = model.transform(df_test)

# 1. Evaluar precisión (accuracy)
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy:.2f}, tiempo: {tiempo_ejecucion_RF:.2f}")

writeJson(tiempo_ejecucion_RF, accuracy, "Random Forest")

# Cerrar la sesión de Spark
spark.stop()
