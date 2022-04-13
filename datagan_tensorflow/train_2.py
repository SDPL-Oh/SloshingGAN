import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from data import LoadTfrecord
from models import Generator, Discriminator
from statistics import WeibullDistribution

mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))


class SloshingGan(tf.keras.Model):
    def __init__(self, gen, disc, latent, global_batch_size):
        super(SloshingGan, self).__init__()
        self.latent = latent
        self.gen = gen
        self.disc = disc
        self.global_batch_size = global_batch_size

    def compile(self, gen_opt, disc_opt, loss_fn):
        super(SloshingGan, self).compile()
        self.gen_opt = gen_opt
        self.disc_opt = disc_opt
        self.loss_fn = loss_fn

    def compute_loss(self, per_example_loss):
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)

    def gen_loss(self, fake_output):
        per_example_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        return self.compute_loss(per_example_loss)

    def disc_loss(self, real_output, fake_output):
        real_loss = self.compute_loss(self.loss_fn(tf.ones_like(real_output), real_output))
        fake_loss = self.compute_loss(self.loss_fn(tf.zeros_like(fake_output), fake_output))
        total_loss = real_loss + fake_loss
        return total_loss, real_loss, fake_loss

    def train_step(self, tf_data):
        index, inputs, outputs = tf_data
        batch = tf.shape(inputs)[0]
        noise = tf.random.normal([batch, self.latent])
        inputs = tf.concat([inputs, noise], -1)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_vectors = self.gen(inputs, training=True)
            real_output = self.disc(outputs, training=True)
            fake_output = self.disc(gen_vectors, training=True)

            gen_loss = self.gen_loss(fake_output)
            disc_loss, real_loss, fake_loss = self.disc_loss(real_output, fake_output)

            grads_gen = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
            grads_disc = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

            self.gen_opt.apply_gradients(
                zip(grads_gen, self.gen.trainable_variables))
            self.disc_opt.apply_gradients(
                zip(grads_disc, self.disc.trainable_variables))
        return {'gen_loss': gen_loss, 'disc_loss': disc_loss, 'real_loss': real_loss, 'fake_loss': fake_loss}

    def test_step(self, tf_data):
        index, inputs, outputs = tf_data
        batch_size = tf.shape(inputs)[0]
        noise = tf.random.normal([batch_size, self.latent])
        inputs = tf.concat([inputs, noise], -1)
        gen_vectors = self.gen(inputs, training=False)
        fake_output = self.disc(gen_vectors, training=True)
        gen_loss = self.gen_loss(fake_output)
        return {"gen_loss": gen_loss}

    def predict_step(self, tf_data):
        index, inputs, outputs = tf_data
        batch_size = tf.shape(inputs)[0]
        noise = tf.random.normal([batch_size, self.latent])
        inputs = tf.concat([inputs, noise], -1)
        predict_vector = self.generator_model(inputs, training=False)
        return {"predict_output": predict_vector, "index": index, "inputs": inputs}


class Algorithm:
    def __init__(self, hparams):
        self.input_size = hparams['input_size']
        self.output_size = hparams['output_size']
        self.latent = hparams['latent']
        self.samples = hparams['samples']
        self.batch = hparams['batch']
        self.epochs = hparams['epochs']
        self.gen_lr = hparams['generator_lr']
        self.disc_lr = hparams['discriminator_lr']
        self.decay_steps = hparams['decay_steps']
        self.decay_rate = hparams['decay_rate']
        self.num_gen_data = hparams['num_gen_data']
        self.model_path = hparams['model_path']
        self.logs_path = hparams['logs_path']
        self.train_data = hparams['train_data']
        self.test_data = hparams['test_data']
        self.input_columns = hparams['input_columns']
        self.target_columns = hparams['target_columns']

        self.global_batch_size = (self.batch * mirrored_strategy.num_replicas_in_sync)

    def callbacks(self):
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            self.logs_path, histogram_freq=100
        )
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            save_weights_only=True,
            verbose=1,
            save_freq=5000)
        return tensorboard_cb, checkpoint_cb

    def train(self):
        next_batch = LoadTfrecord(self.epochs)
        train_dataset = next_batch.get_dataset_next(self.train_data, self.global_batch_size)
        test_dataset = next_batch.get_dataset_next(self.test_data, self.global_batch_size)

        with mirrored_strategy.scope():
            gen = Generator(self.input_size, self.latent, self.output_size)
            disc = Discriminator(self.output_size, 1)
            ###################### load model ######################
            # gen = tf.keras.models.load_model(self.gen_path)
            # disc = tf.keras.models.load_model(self.disc_path)
            ########################################################
            models = SloshingGan(gen, disc, self.latent, self.global_batch_size)

            models.compile(
                gen_opt=tf.keras.optimizers.Adam(learning_rate=self.gen_lr),
                disc_opt=tf.keras.optimizers.Adam(learning_rate=self.disc_lr),
                loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                           reduction=tf.keras.losses.Reduction.NONE)
            )

        models.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=test_dataset,
            callbacks=[self.callbacks()]
        )

        gen.save(self.model_path + 'gen/')
        disc.save(self.model_path + 'disc/')