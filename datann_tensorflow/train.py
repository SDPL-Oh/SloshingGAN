import pandas as pd
import tensorflow as tf
from data import LoadTfrecord
from model import DNN
from fitting import WeibullDistribution

class SloshingNN(tf.keras.Model):
    def __init__(self, logic, batch_size):
        super(SloshingNN, self).__init__()
        self.logic = logic
        self.batch_size = batch_size

    def compile(self, optimizer, loss_fn):
        super(SloshingNN, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def compute_loss(self, labels, predictions):
        per_example_loss = self.loss_fn(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss,
                                          global_batch_size=self.batch_size)

    def train_step(self, tf_data):
        index, inputs, outputs = tf_data
        with tf.GradientTape() as tape:
            predict_vector = self.logic(inputs, training=True)
            loss = self.compute_loss(outputs, predict_vector)
        grads = tape.gradient(loss, self.logic.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.logic.trainable_weights))
        return {"loss": loss}

    def test_step(self, tf_data):
        index, inputs, outputs = tf_data
        predict_vector = self.logic(inputs, training=False)
        loss = self.compute_loss(outputs, predict_vector)
        return {"loss": loss}

    def predict_step(self, tf_data):
        index, inputs, outputs = tf_data
        predict_vector = self.logic(inputs, training=False)
        return {"predict_output": predict_vector,
                "filename": index,
                "target_output": outputs}

class Algorithm:
    def __init__(self, hparams):
        self.input_size = hparams['input_size']
        self.output_size = hparams['output_size']
        self.batch_size = hparams['batch_size']
        self.epochs = hparams['epochs']
        self.lr = hparams['lr']
        self.model_path = hparams['model_path']
        self.logs_path = hparams['logs_path']
        self.train_data = hparams['train_data']
        self.test_data = hparams['test_data']
        self.input_columns = hparams['input_columns']
        self.target_columns = hparams['target_columns']

    def callbacks(self):
        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            self.logs_path, histogram_freq=100
        )
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            save_weights_only=True,
            verbose=1,
            save_freq=20000)
        return tensorboard_cb, checkpoint_cb

    def train(self):
        next_batch = LoadTfrecord(self.epochs)
        train_dataset = next_batch.get_dataset(self.train_data, self.batch_size)
        test_dataset = next_batch.get_dataset(self.test_data, self.batch_size)

        logic = DNN(self.input_size, self.output_size)
        ###################### load model ######################
        # logic = tf.keras.models.load_model(self.model_path)
        ########################################################
        models = SloshingNN(logic, self.batch_size)

        models.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss_fn=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        )
        models.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=test_dataset,
            callbacks=[self.callbacks()]
        )
        logic.save(self.model_path + 'nn/')

    def test(self, csv_file, conditions):
        logic = tf.keras.models.load_model(self.model_path + 'nn/')
        models = SloshingNN(logic, self.batch_size)
        plot_data = WeibullDistribution(self.target_columns)

        data_info = pd.read_csv(csv_file)
        inputs = pd.DataFrame.from_dict(conditions)
        inputs = plot_data.normalized(inputs, data_info, self.input_columns)

        data = models.predict(inputs)['predict_output']
        data = pd.DataFrame(data, columns=self.target_columns)
        data = plot_data.denormalized(data, data_info, self.target_columns)

        raw_data = plot_data.extract_dat(data_info, conditions)
        plot_data.plot_weibull_scipy(data[self.target_columns[0]][:], raw_data, conditions)
