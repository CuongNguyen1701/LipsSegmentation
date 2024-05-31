
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Model
from keras.layers import *
from keras.metrics import *
from hparams import *
from loss_functions import JaccardLoss
class PreprocessingBlock(Layer):
    def __init__(self, image_size=IMAGE_SIZE, scale=1./255):
        super().__init__()
        self.resize = Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1])
        self.rescale = Rescaling(scale=1./255)
    def call(self, x):
        x = self.resize(x)
        x = self.rescale(x)
        return x
class ConvBlock(Layer):
        def __init__(self, filters, kernel_size, strides=1, padding='same', drop_out_rate=0.1, pool=False):
            super().__init__()

            self.conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal')
            self.dropout = Dropout(drop_out_rate)
            self.norm = BatchNormalization()
            self.relu = LeakyReLU(0.1)

        def call(self, x):
            x = self.conv(x)
            x = self.norm(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x


class UpConvBlock(Layer):
    def __init__(self, filters, kernel_size, strides=2, padding='same', drop_out_rate=0.1, use_attention=False):
            super().__init__()
            self.tconv = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal')
            self.dropout = Dropout(drop_out_rate)
            self.norm = BatchNormalization()
            self.relu = ReLU()
            self.concat = Concatenate(axis=3)
            self.use_attention = use_attention
            self.attention = AdditiveAttention()
    def call(self, x, y=None):
            x = self.tconv(x)
            if y != None:
                if self.use_attention:
                    x = self.attention([x, y])
                else:
                    x = self.concat([x, y])
            x = self.norm(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x   
        
def get_model(pretrain_path=None):
    inputs = Input((None, None, 3))

    preprocessed = PreprocessingBlock()(inputs)

    conv1 = ConvBlock(16, 3)(preprocessed)
    conv1 = ConvBlock(16, 3)(conv1) 
    pool1 = MaxPooling2D((2,2))(conv1)

    conv2 = ConvBlock(32, 3)(pool1)
    conv2 = ConvBlock(32, 3)(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)

    conv3 = ConvBlock(64, 3)(pool2)
    conv3 = ConvBlock(64, 3)(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    conv4 = ConvBlock(128, 3)(pool3)
    conv4 = ConvBlock(128, 3)(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)

    conv5 = ConvBlock(256, 3)(pool4)
    conv5 = ConvBlock(256, 3)(conv5)

    upconv6 = UpConvBlock(filters=128, kernel_size=2)(conv5, conv4)
    conv6 = ConvBlock(128, 3)(upconv6)
    conv6 = ConvBlock(128, 3)(conv6)

    upconv7 = UpConvBlock(filters=64, kernel_size=2)(conv6, conv3)
    conv7 = ConvBlock(64, 3)(upconv7)
    conv7 = ConvBlock(64, 3)(conv7)

    upconv8 = UpConvBlock(filters=32, kernel_size=2)(conv7, conv2)
    conv8 = ConvBlock(32, 3)(upconv8)
    conv8 = ConvBlock(32, 3)(conv8)

    upconv9 = UpConvBlock(filters=16, kernel_size=2)(conv8, conv1)
    conv9 = ConvBlock(16, 3)(upconv9)
    conv9 = ConvBlock(16, 3)(conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    # model = tf.keras.Model(inputs=[inputs], outputs=[conv4])
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    metrics = [MeanIoU(num_classes=2)]
    model.compile(optimizer='adam', loss=JaccardLoss, metrics=metrics)
    if(pretrain_path):
        model.load_weights(pretrain_path)
    return model