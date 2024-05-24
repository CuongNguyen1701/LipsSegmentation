
import tensorflow as tf
from tensorflow import keras
import keras.backend as keras_be
def JaccardLoss(targets, inputs, smooth=1e-6):
    # Convert targets to float32 data type
    targets = keras_be.cast(targets, dtype='float32')

    # Flatten label and prediction tensors
    inputs = keras_be.flatten(inputs)
    targets = keras_be.flatten(targets)

    # Reshape flattened tensors to rank-2 matrices
    inputs = keras_be.reshape(inputs, (-1, 1))
    targets = keras_be.reshape(targets, (-1, 1))

    # Calculate intersection and union using dot product
    intersection = keras_be.sum(keras_be.dot(keras_be.transpose(targets), inputs))
    total = keras_be.sum(targets) + keras_be.sum(inputs)
    union = total - intersection

    # Calculate IoU (Intersection over Union)
    IoU = (intersection + smooth) / (union + smooth)

    # Calculate Jaccard loss
    return 1.0 - IoU