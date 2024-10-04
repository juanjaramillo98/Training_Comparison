import tensorflow as tf
import os

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
img_size = 128

def make_dataset(path):
    ds = tf.data.Dataset.list_files(str(path/'*/*'), shuffle=True)
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    return ds

def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_size, img_size])

def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == ["NEUMONIA","NORMAL"]
    # Integer encode the label
    return tf.cast(one_hot, tf.int64)