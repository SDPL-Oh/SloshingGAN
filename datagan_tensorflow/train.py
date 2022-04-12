import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from models import Generator, Discriminator


# TODO: 위치에 대한 센서 번호가 정해지면 적용할 코드
# c = tf.sparse.reshape(example['object/class/label'], [-1])
# c = tf.one_hot(tf.sparse.to_dense(c), depth=10, axis=-1)

class LoadTfrecord:
    def __init__(self, epochs):
        self.epochs = epochs

    def read_tfrecord(self, example):
        tfrecord_format = ({
            'Index': tf.io.FixedLenFeature((), tf.float32),
            'Hs': tf.io.VarLenFeature(tf.float32),
            'Tz': tf.io.VarLenFeature(tf.float32),
            'Speed': tf.io.VarLenFeature(tf.float32),
            'Heading': tf.io.VarLenFeature(tf.float32),
            'Sensor': tf.io.VarLenFeature(tf.float32),
            'Pressure': tf.io.VarLenFeature(tf.float32)
        })
        example = tf.io.parse_single_example(example, tfrecord_format)
        inputs = tf.sparse.concat(axis=0, sp_inputs=[
            example['Hs'],
            example['Tz'],
            example['Speed'],
            example['Heading'],
            example['Sensor']])
        inputs = tf.sparse.to_dense(inputs)
        return example['Index'], inputs, example['Pressure']

    def load_data(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def get_dataset(self, filenames, batch_size, samples, is_training=True):
        dataset = self.load_data(filenames)
        if is_training:
            dataset = dataset.shuffle(samples, reshuffle_each_iteration=False)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).repeat(self.epochs)
        return iter(dataset)


class SloshingGan:
    def __init__(self, hparams):
        self.input_size = hparams['input_size']
        self.output_size = hparams['output_size']
        self.latent = hparams['latent']
        self.samples = hparams['samples']
        self.batch = hparams['batch']
        self.epochs = hparams['epochs']
        self.generator_lr = hparams['generator_lr']
        self.discriminator_lr = hparams['discriminator_lr']
        self.decay_steps = hparams['decay_steps']
        self.decay_rate = hparams['decay_rate']
        self.num_gen_data = hparams['num_gen_data']
        self.model_path = hparams['model_path']
        self.logs_path = hparams['logs_path']
        self.train_data = hparams['train_data']
        self.test_data = hparams['test_data']
        self.input_columns = hparams['input_columns']
        self.target_columns = hparams['target_columns']

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.generator_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True)

        self.generator = Generator(self.input_size, self.latent, self.output_size)
        self.discriminator = Discriminator(self.output_size, 1)
        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.discriminator_opt = tf.keras.optimizers.Adam(learning_rate=self.discriminator_lr)
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_opt,
                                              discriminator_optimizer=self.discriminator_opt,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, self.model_path, max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored.')

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss, real_loss, fake_loss

    @tf.function
    def train_step(self, x, y, batch):
        noise = tf.random.normal([batch, self.latent])
        inputs = tf.concat([x, noise], -1)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_vectors = self.generator(inputs, training=True)
            real_output = self.discriminator(y, training=True)
            fake_output = self.discriminator(generated_vectors, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss, real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_opt.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_opt.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return {'gen_loss': gen_loss, 'disc_loss': disc_loss, 'real_loss': real_loss, 'fake_loss': fake_loss}

    @tf.function
    def predict_step(self, x, batch):
        noise = tf.random.normal([batch, self.latent])
        inputs = tf.concat([x, noise], -1)
        return self.generator_model(inputs, training=False)

    def train(self, profiler):
        logs = self.logs_path + datetime.now().strftime("%Y%m%d-%H%M%S") + '/train'
        total_step = 0
        train_summary_writer = tf.summary.create_file_writer(logs)
        tf.summary.trace_on(graph=True, profiler=profiler)
        next_batch = LoadTfrecord(self.epochs)
        train_dataset = next_batch.get_dataset(self.train_data, self.batch, self.samples)
        print("Starting training")
        for epoch in range(self.epochs):
            print("Start epoch {} of training".format(epoch))
            for step in range(int(self.samples / self.batch)):
                index, inputs, outputs = next(train_dataset)
                step_pre_batch = len(index)
                batch_logs = self.train_step(inputs, outputs, step_pre_batch)

                if epoch == 0 and step == 20:
                    with train_summary_writer.as_default():
                        tf.summary.trace_export(
                            name="algorithm_trace",
                            step=epoch,
                            profiler_outdir=logs)

                if step % 200 == 0:
                    print("step %d ---- G: %.6f ---- D: %.6f ---- D(r): %.6f ---- D(f): %.6f" %
                          (step, float(batch_logs['gen_loss']), float(batch_logs['disc_loss']),
                           float(batch_logs['real_loss']), float(batch_logs['fake_loss'])))
                    with train_summary_writer.as_default():
                        tf.summary.scalar('gen_loss', batch_logs['gen_loss'], step=total_step)
                        tf.summary.scalar('disc_loss', batch_logs['disc_loss'], step=total_step)
                        tf.summary.scalar('real_loss', batch_logs['real_loss'], step=total_step)
                        tf.summary.scalar('fake_loss', batch_logs['fake_loss'], step=total_step)
                        for gv in self.generator.trainable_variables:
                            tf.summary.histogram(gv.name, gv, step=total_step)
                        for dv in self.discriminator.trainable_variables:
                            tf.summary.histogram(dv.name, dv, step=total_step)

                    train_summary_writer.flush()
                    total_step += 200

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

            if (epoch + 1) % 15 == 0:
                model_save_path = self.model_path + 'models'
                self.generator.save(model_save_path)
                print('Saving Model for epoch {} at {}'.format(epoch, model_save_path))

    def test(self, csv_file, inputs):
        data_list = []
        data_info = pd.read_csv(csv_file)

        inputs = pd.DataFrame.from_dict(inputs)
        inputs = self.normalized(inputs, data_info, self.input_columns)
        self.generator_model = tf.keras.models.load_model(self.model_path + 'models')

        for _ in tqdm(range(self.num_gen_data), desc="Generate Data"):
            data = self.predict_step(inputs, 1)
            data = self.denormalized(data, data_info, self.target_columns)
            data_list.append(np.array(data)[0])

        plt.scatter(data_list,
                    stats.exponweib.pdf(data_list, *stats.exponweib.fit(data_list, 1, 1, scale=2, loc=0)),
                    label='gen')
        plt.scatter(data_info[self.target_columns][:62],
                    stats.exponweib.pdf(data_info[self.target_columns][:62],
                                        *stats.exponweib.fit(data_info[self.target_columns][:62],
                                                             1, 1, scale=2, loc=0)),
                    label='exp')
        plt.legend()
        plt.show()

    def normalized(self, data, standard_data, columns):
        data_stats = standard_data.describe()
        data_stats = data_stats.transpose()
        trans_data = (data - data_stats['min']) / ((data_stats['max']) - data_stats['min'])
        trans_data = trans_data[columns].fillna(0)
        return trans_data.astype('float32')

    def denormalized(self, data, standard_data, columns):
        data_stats = standard_data.describe()
        data_stats = data_stats.transpose()
        trans_data = (data * ((data_stats['max'][columns]) - data_stats['min'][columns])) + data_stats['min'][columns]
        return trans_data