import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from data import LoadTfrecord
from models import Generator, Discriminator
from statistics import WeibullDistribution


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

            if (epoch + 1) % 2 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

            if (epoch + 1) % 5 == 0:
                model_save_path = self.model_path + 'models'
                self.generator.save(model_save_path)
                print('Saving Model for epoch {} at {}'.format(epoch, model_save_path))

    def test(self, csv_file, inputs):
        plot_data = WeibullDistribution(self.target_columns)
        data_list = []
        data_info = pd.read_csv(csv_file)

        inputs = pd.DataFrame.from_dict(inputs)
        inputs = plot_data.normalized(inputs, data_info, self.input_columns)
        self.generator_model = tf.keras.models.load_model(self.model_path + 'models')

        for _ in tqdm(range(self.num_gen_data), desc="Generate Data"):
            data = self.predict_step(inputs, 1)
            data = plot_data.denormalized(data, data_info, self.target_columns)
            data_list.append(np.array(data)[0])

        plot_data.plot_scatter(data_list, data_info)



