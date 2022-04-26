import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self, input_size, latent, output_size):
        super().__init__()
        self.logic = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(input_size+latent,)),
                tf.keras.layers.Dense(25),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(15),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(output_size)
            ],
            name="Generator",
        )

    def call(self, inputs):
        return self.logic(inputs)


class Discriminator(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.logic = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(input_size,)),
                tf.keras.layers.Dense(10, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(10),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(output_size)
            ],
            name="Discriminator",
        )

    def call(self, inputs):
        return self.logic(inputs)