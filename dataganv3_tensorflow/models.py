import os
import tensorflow as tf
import tensorflow_addons as tfa
from dataganv3_tensorflow import probability as pb


def instance_norm(epsilon=1e-5):
    return tfa.layers.normalizations.InstanceNormalization(
        epsilon=epsilon,
        scale=True,
        center=True
    )


def leaky_relu(x=None, alpha=0.01):
    if x is None:
        return tf.keras.layers.LeakyReLU(alpha=alpha)
    else:
        return tf.keras.layers.LeakyReLU(alpha=alpha)(x)


def regularization_loss(model):
    loss = tf.nn.scale_regularization_loss(model.losses)
    return loss


def generator_loss(fake_output):
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake_output),
            logits=fake_output
        )
    )
    return fake_loss


def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real_output),
            logits=real_output
        )
    )
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake_output),
            logits=fake_output
        )
    )
    return real_loss + fake_loss


def l1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def weibull_loss(x, y):
    shape, scale, location = y
    data_mean = tf.math.reduce_mean(x)
    data_var = tf.math.reduce_variance(x)
    data_skew = tf.math.divide(data_mean, data_var)
    cdf_skew, cdf_var, cdf_mean = pb.estimate_para_cal(shape, scale, location)

    skew_loss = tf.keras.losses.mean_absolute_error(cdf_skew, data_skew)
    location_loss = tf.keras.losses.mean_absolute_error(cdf_var, data_var)
    shape_loss = tf.keras.losses.mean_absolute_error(cdf_mean, data_mean)

    return shape_loss, location_loss, skew_loss


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def flatten(x=None, name='flatten'):
    if x is None:
        return tf.keras.layers.Flatten(name=name)
    else:
        return tf.keras.layers.Flatten(name=name)(x)


def down_sample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters, size, strides=2, padding='same',
            use_bias=False))
    if apply_batchnorm:
        result.add(instance_norm())
    result.add(leaky_relu(alpha=0.2))

    return result


def up_sample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2, padding='same',
            use_bias=False))
    result.add(instance_norm())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(leaky_relu(alpha=0.2))

    return result


class Discriminator(tf.keras.Model):
    def __init__(self, size):
        super().__init__()
        self.logic = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=[size, size, 1]),
                tf.keras.layers.Conv2D(
                    64, 3, strides=1,
                    use_bias=False),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(
                    128, 3, strides=1,
                    use_bias=False),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(
                    256, 3, strides=1,
                    use_bias=False)
            ]
        )
        self.logic.summary()

    def call(self, inputs, training=True, mask=None):
        x = self.logic(inputs)
        return x


def discriminator_model(size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[size, size, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


def make_generator_model(size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def generator(size, channel):
    inputs = tf.keras.layers.Input(shape=[size, size, channel])

    down_stack = [
        down_sample(64, 4, apply_batchnorm=False),
        down_sample(128, 4),
        down_sample(256, 4),
        down_sample(512, 4),
    ]

    up_stack = [
        up_sample(256, 4, apply_dropout=True),
        up_sample(128, 4),
        up_sample(64, 4),
        up_sample(32, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
