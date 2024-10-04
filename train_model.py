import tensorflow as tf
import numpy as np
import pathlib
import os
import time
import json
from datetime import datetime, timedelta
from dependencias.modelcompile import train

try:
    from spark_tensorflow_distributor import MirroredStrategyRunner # type: ignore
    cluster_runner = True
except:
    cluster_runner = False
    print("No esta MirroredStrategyRunner disponible")

gpus = len(tf.config.list_physical_devices('GPU'))

if gpus >= 1:
    gpu_runner = True
    print("Num GPUs Available: ", gpus)
else:
    gpu_runner = False
    print("No hay Gpus Disponibles")


localPath = pathlib.Path().resolve()

train_data_path = pathlib.Path(localPath ,"Dataset/TRAIN")
test_data_path = pathlib.Path(localPath ,"Dataset/TEST")
val_data_path = pathlib.Path(localPath ,"Dataset/VALIDATION")


class_names = np.array(sorted([item.name for item in train_data_path.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
useCallbacks = False
BUFFER_SIZE = 100000
Epochs = 1
Steps_per_epoch = 1
Slots = 2

params = {
    "trainPath" : train_data_path,
    "valPath" : val_data_path,
    "epochs" : Epochs,
    "stepsPerEpoch" : Steps_per_epoch,
    "useCallbacks" : useCallbacks
}

inicio = time.time()

if cluster_runner:
    ejecucion = MirroredStrategyRunner(
        num_slots=Slots,
        use_gpu = False,
        local_mode = False,
        use_custom_strategy=True
    )
    history,modelo_final = ejecucion.run(train,kwargs = params)

    print("Se termino la ejecucion del entrenamiento en paralelo")
else:
    history,modelo_final = train(**params)
    print("Se termino la ejecucion del entrenamiento normal")

fin = time.time()


# Calcular el tiempo transcurrido
tiempo_ejecucion = fin - inicio

if gpu_runner :
    tipo_ejecucion = "Ejecucion con GPU"
    modelo_final.save('modelos/modeloGPU.keras')
    Slots = 0
elif cluster_runner:
    tipo_ejecucion = "Ejecucion con Cluster"
    modelo_final.save('modelos/modeloCluster.keras')
else:
    tipo_ejecucion = "Ejecucion con CPU"
    modelo_final.save('modelos/modeloCPU.keras')
    Slots = 0

# Crear un nuevo registro
nuevo_registro = {
    'Tipo Ejecucion' : tipo_ejecucion,
    'timestamp': (datetime.utcnow() - timedelta(hours=5)).isoformat(),
    'Steps_per_epoch': Steps_per_epoch,
    'Epochs':Epochs,
    'tiempo_ejecucion': tiempo_ejecucion,
    'tiempo_ejecucion_mins': tiempo_ejecucion/60,
    'tiempo_ejecucion_horas': tiempo_ejecucion/3600,
    'Slots' : Slots,
    'accuracy' : history.history['accuracy']
}

# Nombre del archivo JSON
nombre_archivo = 'tiempo_ejecucion.json'

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

print(f"Nuevo registro guardado en '{nombre_archivo}'")
