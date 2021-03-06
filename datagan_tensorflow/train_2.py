import numpy as np
import os
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
        inputs = tf_data
        batch_size = tf.shape(inputs)[0]
        noise = tf.random.normal([batch_size, self.latent])
        inputs = tf.concat([inputs, noise], -1)
        predict_vector = self.gen(inputs, training=False)
        return {"predict_output": predict_vector}

class CheckCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint, model_path, num_epoch):
        self.model_path = model_path
        self.checkpoint = checkpoint
        self.num_epoch = num_epoch

    def on_train_begin(self, logs=None):
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, self.model_path, max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored.')

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.num_epoch == 0:
            self.ckpt_save_path = self.ckpt_manager.save()
            print('\n saving model to {}'.format(self.model_path))


class Algorithm:
    def __init__(self, hparams):
        self.input_size = len(hparams['input_columns'])
        self.output_size = len(hparams['target_columns'])
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
            self.logs_path + datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=100
        )
        return tensorboard_cb

    def train(self):
        next_batch = LoadTfrecord(self.epochs, self.input_columns)
        train_dataset = next_batch.get_dataset_next(self.train_data, self.global_batch_size)
        test_dataset = next_batch.get_dataset_next(self.test_data, self.global_batch_size)


        with mirrored_strategy.scope():
            gen = Generator(self.input_size, self.latent, self.output_size)
            disc = Discriminator(self.output_size, 1)
            gen_opt = tf.keras.optimizers.Adam(learning_rate=self.gen_lr)
            disc_opt = tf.keras.optimizers.Adam(learning_rate=self.gen_lr)
            checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                             discriminator_optimizer=disc_opt,
                                             generator=gen,
                                             discriminator=disc)
            ########################################################
            models = SloshingGan(gen, disc, self.latent, self.global_batch_size)

            models.compile(
                gen_opt=gen_opt,
                disc_opt=disc_opt,
                loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                           reduction=tf.keras.losses.Reduction.NONE)
            )

        models.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=test_dataset,
            callbacks=[self.callbacks(), CheckCallback(checkpoint, self.model_path, 5)]
        )

        gen.save(self.model_path + 'gen/')
        disc.save(self.model_path + 'disc/')

    def test(self, csv_file,  conditions):
        gen = tf.keras.models.load_model(self.model_path + 'gen/')
        disc = tf.keras.models.load_model(self.model_path + 'disc/')
        models = SloshingGan(gen, disc, self.latent, self.num_gen_data)
        plot_data = WeibullDistribution(self.target_columns)

        inputs_list = pd.DataFrame()
        data_info = pd.read_csv(csv_file)
        if isinstance(conditions, dict):
            inputs = pd.DataFrame.from_dict(conditions)
        else:
            inputs = pd.read_csv(conditions)

        inputs_norm = plot_data.normalized(inputs, data_info, self.input_columns)
        for dat_idx in range(len(inputs_norm)):
            for idx in range(self.num_gen_data):
                inputs_list = pd.concat([inputs_list, inputs_norm.iloc[dat_idx:dat_idx+1]], ignore_index=True)
        print(inputs_list.tail())

        data = models.predict(inputs_list)['predict_output']
        data = pd.DataFrame(data, columns=self.target_columns)
        data = plot_data.denormalized(data, data_info, self.target_columns)

        p_df = pd.DataFrame()
        for dat_idx in tqdm(range(len(inputs))):
            raw_data = plot_data.extract_dat(data_info,
                                             inputs.iloc[dat_idx:dat_idx+1].reset_index(),
                                             self.input_columns)
            try:
                end_idx = dat_idx + 1
                if end_idx == len(inputs):
                    end_idx = -1
                parameter = plot_data.plot_weibull_scipy(
                    data[self.target_columns[0]][self.num_gen_data*dat_idx:self.num_gen_data*(end_idx)],
                    raw_data,
                    inputs.iloc[dat_idx:dat_idx+1],
                    self.logs_path + 'plt/{}'.format(dat_idx))

                p_df = pd.concat(
                    [p_df,
                     pd.DataFrame([parameter], columns=['p_shape', 'p_loc', 'p_scale', 'r_shape', 'r_loc', 'r_scale'])
                     ])
            except:
                pass

        p_df.to_csv(self.logs_path + 'plt/{}'.format(datetime.now().strftime("%Y%m%d.csv")), index=False)





