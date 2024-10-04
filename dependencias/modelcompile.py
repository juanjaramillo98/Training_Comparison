from .datasetloader import make_dataset

import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import  Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.applications.vgg19 import VGG19 # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore


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

    
    es = EarlyStopping(monitor="val_loss", verbose=1, mode="min", patience=4)
    cp=ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, save_weights_only=False,mode="auto", save_freq="epoch")
    lrr = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=0, factor=0.5, min_lr=0.0001)

    callbacks=[es,cp,lrr]

    sgd = SGD(learning_rate=0.0001, decay = 1e-6, momentum=1e-6, nesterov = True)

    model_01.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
    return model_01,callbacks

def train(trainPath,valPath,epochs,stepsPerEpoch,useCallbacks):
    train_datasets = make_dataset(trainPath)
    validation_data= make_dataset(valPath)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_datasets = train_datasets.with_options(options)
    multi_worker_model,callbacks = build_and_compile_VGG19()
    history = multi_worker_model.fit(
        x=train_datasets, 
        epochs=epochs, 
        steps_per_epoch=stepsPerEpoch,
        validation_data=validation_data,
        callbacks=callbacks if useCallbacks else []
    )
    return history,multi_worker_model