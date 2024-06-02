import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Unet:
    def __init__(self):
        return
        
    def double_conv_block(self, x, n_filters):
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        return x

    def downsample_block(self, x, n_filters):
        f = self.double_conv_block(x, n_filters)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(0.3)(p)
        return f, p

    def upsample_block(self, x, conv_features, n_filters):
        x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        x = layers.concatenate([x, conv_features])
        x = layers.Dropout(0.3)(x)
        x = self.double_conv_block(x, n_filters)
        return x

    def build_unet_model(self):
        inputs = layers.Input(shape=(256,256,3))

        # encoder: contracting path - downsample
        f1, p1 = self.downsample_block(inputs, 64)
        f2, p2 = self.downsample_block(p1, 128)
        f3, p3 = self.downsample_block(p2, 256)
        f4, p4 = self.downsample_block(p3, 512)

        # bottleneck
        bottleneck = self.double_conv_block(p4, 1024)

        # decoder: expanding path - upsample
        u6 = self.upsample_block(bottleneck, f4, 512)
        u7 = self.upsample_block(u6, f3, 256)
        u8 = self.upsample_block(u7, f2, 128)
        u9 = self.upsample_block(u8, f1, 64)

        # outputs
        outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)

        # unet model with Keras Functional API
        unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

        return unet_model
    