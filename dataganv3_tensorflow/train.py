import time
import pandas as pd
from datetime import datetime
from scipy import stats
from dataganv3_tensorflow.data import *
from dataganv3_tensorflow.models import *
from dataganv3_tensorflow.probability import *

mirrored_strategy = tf.distribute.MirroredStrategy()
tf.config.run_functions_eagerly(True)


def conclusion():
    result_file = pd.read_csv('F:/SloshingProject/data/osj/paper/result_model3_64_10.csv')
    normal_distribution(result_file, 'shape', 'shape_3_64_10.jpg')


class SloshingGAN:
    def __init__(self):
        super(SloshingGAN, self).__init__()
        self.model_name = 'datagan_v3'
        self.phase = 'train'
        self.checkpoint_dir = 'checkpoint'

        self.tfrecord = "F:/SloshingProject/data/osj/train_parameters_64.record"
        self.input_columns = ['hs', 'tz', 'speed', 'heading', 'loc', 's_shape', 's_scale', 's_location']
        self.target_columns = ['pressure']

        self.size = 64
        self.batch_size = 1
        self.iteration = 1001
        self.save_freq = 1000
        self.lr = 0.0003
        self.latent_dim = 10

        self.checkpoint_dir = os.path.join('checkpoint', self.model_dir)
        self.log_dir = os.path.join('logs', self.model_dir)
        self.asset_dir = os.path.join('assets', self.model_dir, datetime.now().strftime("%Y%m%d"))
        check_folder(self.checkpoint_dir)
        check_folder(self.log_dir)
        check_folder(self.asset_dir)

        print()
        print("##### Information #####")
        print("# batch_size : ", self.batch_size)
        print("# max iteration : ", self.iteration)
        print()
        print("##### Generator #####")
        print("# latent_dim : ", self.latent_dim)
        print()

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        iter_data = LoadTfrecord(self.size, input_channel=len(self.input_columns))
        self.generator = generator(self.size, self.latent_dim + len(self.input_columns))
        self.generator.summary()

        if self.phase == 'train':
            dataset = iter_data.get_dataset_next(self.tfrecord, self.batch_size)
            self.dataset_iter = iter(dataset)
            # self.discriminator = Discriminator(self.size)
            self.discriminator = discriminator_model(self.size)
            self.discriminator.summary()
            self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.0, beta_2=0.99, epsilon=1e-08)
            self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.0, beta_2=0.99, epsilon=1e-08)

            self.ckpt = tf.train.Checkpoint(
                generator=self.generator,
                discriminator=self.discriminator,
                g_optimizer=self.g_optimizer,
                d_optimizer=self.d_optimizer)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=1)
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')

        else:
            dataset = iter_data.get_test_dataset(self.tfrecord)
            self.dataset_iter = iter(dataset)
            self.ckpt = tf.train.Checkpoint(generator=self.generator)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=1)

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored')
            else:
                print('Not restoring from saved checkpoint')

    @tf.function
    def g_test_step(self, x):
        return self.generator(x)

    @tf.function
    def train_step(self, x, y, parameters):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            y_fake = self.generator(x)
            real_logit = self.discriminator(y)
            fake_logit = self.discriminator(y_fake)
            g_adv_loss = generator_loss(fake_logit)
            shape_loss, location_loss, skew_loss = weibull_loss(x, parameters)
            g_wei_loss = tf.cast(tf.reduce_sum([skew_loss, location_loss, shape_loss]), tf.float32)

            g_loss = g_adv_loss + g_wei_loss
            d_adv_loss = discriminator_loss(real_logit, fake_logit)

        g_train_variable = self.generator.trainable_variables
        g_gradient = g_tape.gradient(g_loss, g_train_variable)
        self.g_optimizer.apply_gradients(zip(g_gradient, g_train_variable))

        d_train_variable = self.discriminator.trainable_variables
        d_gradient = d_tape.gradient(d_adv_loss, d_train_variable)
        self.d_optimizer.apply_gradients(zip(d_gradient, d_train_variable))
        return shape_loss, location_loss, skew_loss, g_adv_loss, g_loss, d_adv_loss, y_fake

    def train(self):
        self.build_model()
        logs = self.log_dir + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        train_summary_writer = tf.summary.create_file_writer(logs)
        start_time = time.time()
        tf.summary.trace_on(graph=True, profiler=False)
        for i in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            idx, shape, loc, scale, require_matrix = next(self.dataset_iter)
            noise = tf.random.normal((self.batch_size, self.size, self.size, self.latent_dim))
            input_data = tf.concat([require_matrix, noise], axis=3)

            noise_batches = 0
            for idx_batch in range(self.batch_size):
                noise = stats.weibull_min.rvs(
                    shape.numpy()[idx_batch],
                    loc=loc.numpy()[idx_batch],
                    scale=scale.numpy()[idx_batch],
                    size=self.size * self.size,
                    random_state=None
                )
                if idx_batch == 0:
                    noise_batches = tf.reshape(tf.sort(noise), (1, self.size, self.size, 1))
                else:
                    noise = tf.reshape(tf.sort(noise), (1, self.size, self.size, 1))
                    noise_batches = tf.concat([noise_batches, noise], 0)

            shape_loss, location_loss, skew_loss, g_adv_loss, g_loss, d_adv_loss, y_fake = \
                self.train_step(input_data, noise_batches, [shape, loc, scale])

            with train_summary_writer.as_default():
                tf.summary.scalar('g/shape_loss', shape_loss, step=i)
                tf.summary.scalar('g/location_loss', location_loss, step=i)
                tf.summary.scalar('g/skew_loss', skew_loss, step=i)
                tf.summary.scalar('g/g_adv_loss', g_adv_loss, step=i)
                tf.summary.scalar('g/g_loss', g_loss, step=i)
                tf.summary.scalar('d/adv_loss', d_adv_loss, step=i)

            if np.mod(i + 1, self.save_freq) == 0:
                self.manager.save(checkpoint_number=i + 1)

            if np.mod(i + 1, 100) == 0:
                with train_summary_writer.as_default():
                    for g_v in self.generator.trainable_variables:
                        tf.summary.histogram(g_v.name, g_v, step=i)
                    for d_v in self.discriminator.trainable_variables:
                        tf.summary.histogram(d_v.name, d_v, step=i)

                print("iter: [%6d/%6d] time: %4.4f d_loss: %.4f, g_loss: %.4f, g_adv_loss: %.4f, shape_loss: %.4f, "
                      "location_loss: %.4f, skew_loss: %.4f" %
                      (i + 1, self.iteration, time.time() - iter_start_time, d_adv_loss, g_loss, g_adv_loss,
                       shape_loss, location_loss, skew_loss))

            if np.mod(i + 1, 1000) == 0:
                target_y = tf.reshape(noise_batches.numpy()[0], -1)
                predict_y = tf.reshape(y_fake.numpy()[0], -1)
                plot_weibull(
                    predict_y,
                    target_y,
                    self.asset_dir + '/{}.jpg'.format(i + 1)
                )

        with train_summary_writer.as_default():
            tf.summary.trace_export(
                name="graph",
                step=0,
                profiler_outdir=logs)
        self.manager.save(checkpoint_number=self.iteration)
        print("Total train time: %4.4f" % (time.time() - start_time))

    def test(self):
        self.build_model()

        result_para = pd.DataFrame([], columns=['Index', 'p_shape', 'p_location', 'p_scale'])
        for i in tqdm(range(156)):
            idx, shape, loc, scale, require_matrix = next(self.dataset_iter)
            noise = tf.random.normal((1, self.size, self.size, self.latent_dim))
            require_matrix = tf.expand_dims(require_matrix, axis=0)
            input_data = tf.concat([require_matrix, noise], axis=3)

            noise = stats.weibull_min.rvs(
                shape.numpy(),
                loc=loc.numpy(),
                scale=scale.numpy(),
                size=self.size * self.size,
                random_state=None
            )

            noise_batches = tf.reshape(noise, (1, self.size, self.size, 1))
            y_fake = self.g_test_step(input_data)

            predict_y = tf.reshape(y_fake.numpy(), -1)
            target_y = tf.reshape(noise_batches.numpy(), -1)
            plot_weibull(
                predict_y,
                target_y,
                self.asset_dir + '/{}.jpg'.format(idx.numpy())
            )

            shape, scale, location = estimate_para_scipy(predict_y)
            result_para.loc[i] = [int(idx.numpy()), shape, scale, location]
        result_para.to_csv(self.asset_dir + '/result.csv')

    @property
    def model_dir(self):
        return "{}".format(self.model_name)


def main():
    algorithm = SloshingGAN()
    algorithm.train()
    # algorithm.test()
    # conclusion()


if __name__ == "__main__":
    main()