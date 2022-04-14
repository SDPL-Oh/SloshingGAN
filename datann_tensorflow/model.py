import tensorflow as tf

class DNN(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.logic = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(input_size, )),
                tf.keras.layers.Dense(160),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(160),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(160),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dense(output_size, activation=None)
            ],
            name="Densenet",
        )

    def call(self, inputs):
        return self.logic(inputs)
