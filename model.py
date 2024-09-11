import tensorflow as tf
from tensorflow.keras import layers, Model


class UNet:
    def __init__(self, input_shape=(256, 256, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.build_model()

    def double_conv_block(self, x, n_filters):
        x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
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

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)

        # Downsampling path
        f1, p1 = self.downsample_block(inputs, 64)
        f2, p2 = self.downsample_block(p1, 128)
        f3, p3 = self.downsample_block(p2, 256)
        f4, p4 = self.downsample_block(p3, 512)

        # Bottleneck
        bottleneck = self.double_conv_block(p4, 1024)

        # Upsampling path
        u6 = self.upsample_block(bottleneck, f4, 512)
        u7 = self.upsample_block(u6, f3, 256)
        u8 = self.upsample_block(u7, f2, 128)
        u9 = self.upsample_block(u8, f1, 64)

        # Output layer
        outputs = layers.Conv2D(self.num_classes, 1, activation="softmax")(u9)

        # Model
        model = Model(inputs, outputs)
        return model

