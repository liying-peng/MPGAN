import tensorflow as tf
from tf_models.generator import Generator

keras = tf.keras
layers = keras.layers


class Discriminator(keras.Model):

    def __init__(self, name, kernel_initializer, norm_type='instance'):
        super(Discriminator, self).__init__(name=name)
        constant_initializer = keras.initializers.Constant(0)

        self.c64_conv = layers.Conv3D(64, 4, strides=(2, 2, 2), padding='same', kernel_initializer=kernel_initializer)

        self.c128_conv = layers.Conv3D(128, 4, strides=(2, 2, 2), padding='same', kernel_initializer=kernel_initializer)
        self.c128_norm = Generator.norm(128, norm_type)

        self.c256_conv = layers.Conv3D(256, 4, strides=(2, 2, 2), padding='same', kernel_initializer=kernel_initializer)
        self.c256_norm = Generator.norm(256, norm_type)

        self.c512_conv = layers.Conv3D(512, 4, strides=(2, 2, 2), padding='same', kernel_initializer=kernel_initializer)
        self.c512_norm = Generator.norm(512, norm_type)

        self.last_conv = layers.Conv3D(1, 4, strides=(1, 1, 1), padding='same', kernel_initializer=kernel_initializer,
                                         bias_initializer=constant_initializer)

    def call(self, targets):
        # add noise
        # targets = targets + tf.random_normal(shape=tf.shape(targets), mean=0.0, stddev=0.1)

        x = self.c64_conv(targets)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.c128_conv(x)
        x = self.c128_norm(x)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.c256_conv(x)
        x = self.c256_norm(x)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.c512_conv(x)
        x = self.c512_norm(x)
        x = tf.nn.leaky_relu(x, 0.2)

        x = self.last_conv(x)

        return tf.sigmoid(x)



