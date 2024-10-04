import tensorflow as tf
import numpy as np
import pathlib
import os
import time
import json
from datetime import datetime,timezone,timedelta

from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import  Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.applications.vgg19 import VGG19 # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore

try:
    from spark_tensorflow_distributor import MirroredStrategyRunner # type: ignore
    ClusterRunner = True
except:
    ClusterRunner = False
    print("No esta MirroredStrategyRunner disponible")

gpus = len(tf.config.list_physical_devices('GPU'))

if gpus >= 1:
    GpuRunner = True
    print("Num GPUs Available: ", gpus)
else:
    GpuRunner = False
    print("No hay Gpus Disponibles")


localPath = pathlib.Path().resolve()

train_data_path = pathlib.Path(localPath ,"Dataset/TRAIN")
test_data_path = pathlib.Path(localPath ,"Dataset/TEST")
val_data_path = pathlib.Path(localPath ,"Dataset/VALIDATION")


class_names = np.array(sorted([item.name for item in train_data_path.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

img_size = 128

BUFFER_SIZE = 100000
BATCH_SIZE = 64

Epochs = 1
Steps_per_epoch = 1
Slots = 2

def make_dataset(path):
    ds = tf.data.Dataset.list_files(str(path/'*/*'), shuffle=True)
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    return ds

def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.cast(one_hot, tf.int64)

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_size, img_size])

def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def build_and_compile_VGG19():
    base_model = VGG19(input_shape = (128,128,3),
                        include_top = False,
                        weights = 'imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    flat = Flatten()(x)


    class_1 = Dense(4608, activation = 'relu')(flat)
    dropout = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation = 'relu')(dropout)
    output = Dense(2, activation = 'softmax')(class_2)

    model_01 = Model(base_model.inputs, output)
    #model_01.summary()

    filepath = "model.keras"

    callbacks=[]
    es = EarlyStopping(monitor="val_loss", verbose=1, mode="min", patience=4)
    callbacks.append(es)
    cp=ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, save_weights_only=False,mode="auto", save_freq="epoch")
    callbacks.append(cp)
    lrr = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.0001)
    callbacks.append(lrr)

    sgd = SGD(learning_rate=0.0001, decay = 1e-6, momentum=1e-6, nesterov = True)

    model_01.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
    return model_01,callbacks

def train():
    train_datasets = make_dataset(train_data_path)
    validation_data= make_dataset(val_data_path)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_datasets = train_datasets.with_options(options)
    multi_worker_model,callbacks = build_and_compile_VGG19()
    multi_worker_model.fit(
        x=train_datasets, 
        epochs=Epochs, 
        steps_per_epoch=Steps_per_epoch,
        validation_data=validation_data,
        callbacks=callbacks
    )


inicio = time.time()

if ClusterRunner:
    MirroredStrategyRunner(
        num_slots=Slots,
        use_gpu = False,
        local_mode = False,
        use_custom_strategy=True
        ).run(train)
    print("Se termino la ejecucion del entrenamiento en paralelo")
else:
    train()
    print("Se termino la ejecucion del entrenamiento normal")

fin = time.time()

# Calcular el tiempo transcurrido
tiempo_ejecucion = fin - inicio

if GpuRunner :
    tipoEjecucion = "Ejecucion con GPU"
    Slots = 0
elif ClusterRunner:
    tipoEjecucion = "Ejecucion con Cluster"
else:
    tipoEjecucion = "Ejecucion con CPU"
    Slots = 0

# Crear un nuevo registro
nuevo_registro = {
    'Tipo Ejecucion' : tipoEjecucion,
    'timestamp': (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(),
    'Steps_per_epoch': Steps_per_epoch,
    'Epochs':Epochs,
    'tiempo_ejecucion': tiempo_ejecucion,
    'tiempo_ejecucion_mins': tiempo_ejecucion/60,
    'tiempo_ejecucion_horas': tiempo_ejecucion/3600,
    'Slots' : Slots

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

exit(0)