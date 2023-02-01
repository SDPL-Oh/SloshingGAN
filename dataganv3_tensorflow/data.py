import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

from dataganv3_tensorflow import probability as pb


def normalization_std(input_data, file_name, load=False):
    scaler = StandardScaler()
    if load:
        scaler = joblib.load('./{}.pkl'.format(file_name))
        scale_data = scaler.transform(input_data)
    else:
        scale_data = scaler.fit_transform(input_data)
        joblib.dump(scaler, './{}.pkl'.format(file_name))
    return scale_data


def denormalized(input_data, file_name):
    scaler = joblib.load('./{}.pkl'.format(file_name))
    scale_data = scaler.inverse_transform(input_data)
    return scale_data


def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


class GenerateCSV:
    def __init__(self, pressure_path, scaling_pressure_path, requirements_path, save_path):
        self.pressure_path = pressure_path
        self.scaling_pressure_path = scaling_pressure_path
        self.requirements_path = requirements_path
        self.save_path = save_path
        self.input_columns = ['hs', 'tz', 'speed', 'heading', 'loc']

    def merge_data(self):
        pressures = pd.read_csv(self.pressure_path)
        scaling_pressures = pd.read_csv(self.scaling_pressure_path)
        requirements = pd.read_csv(self.requirements_path)

        para_list = pd.DataFrame([], columns=["t_shape", "t_scale", "t_location", "s_shape", "s_scale", "s_location"])
        for requirement in tqdm(requirements.itertuples(), total=len(requirements)):
            raw_data = self.extract_data(
                pressures,
                requirements.iloc[requirement.Index:requirement.Index + 1].reset_index()
            )

            scaling_data = self.extract_data(
                scaling_pressures,
                requirements.iloc[requirement.Index:requirement.Index + 1].reset_index()
            )

            if raw_data is not None and scaling_data is not None:
                t_shape, t_scale, t_location = pb.estimate_para_scipy(raw_data)
                s_shape, s_scale, s_location = pb.estimate_para_scipy(scaling_data)

                para_list.loc[requirement.Index] = [t_shape, t_scale, t_location, s_shape, s_scale, s_location]

        df = pd.concat([requirements, para_list], axis=1)
        df.to_csv(self.save_path)

    def extract_data(self, df, requirements):
        for col in self.input_columns:
            required = (df[col] == requirements[col][0])
            df = df[required]
        if df.empty:
            return None
        else:
            return df['pressure']


class GenerateTfrecord:
    def __init__(self, requirements_path, save_dir, size):
        self.requirements_path = requirements_path
        self.save_dir = save_dir
        self.size = size
        self.input_columns = ['hs', 'tz', 'speed', 'heading', 'loc']

    def create_requirements(self, requirement):
        metrix_require = np.concatenate(
            (
                np.full(self.size * self.size, requirement.hs),
                np.full(self.size * self.size, requirement.tz),
                np.full(self.size * self.size, requirement.speed),
                np.full(self.size * self.size, requirement.heading),
                np.full(self.size * self.size, requirement.loc),
                np.full(self.size * self.size, requirement.s_shape),
                np.full(self.size * self.size, requirement.s_scale),
                np.full(self.size * self.size, requirement.s_location)
            ),
            dtype=np.float32).tobytes()

        label_dict = {
            'Index': int64_feature(requirement.Index),
            'requirements': bytes_feature(metrix_require),
            't_shape': float_feature(requirement.t_shape),
            't_location': float_feature(requirement.t_location),
            't_scale': float_feature(requirement.t_scale),
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=label_dict))
        return tf_example

    def create_tfrecord(self):
        writer = tf.io.TFRecordWriter(self.save_dir)
        requirements = pd.read_csv(self.requirements_path)
        requirements[self.input_columns] = normalization_std(requirements[self.input_columns], 'requirements')

        for requirement in tqdm(requirements.itertuples(), total=len(requirements)):
            if np.isnan(requirement.s_shape):
                pass
            else:
                tf_example = self.create_requirements(requirement)
                writer.write(tf_example.SerializeToString())

        writer.close()
        print('Successfully created the TFRecord(weibull parameters)')


class LoadTfrecord:
    def __init__(self, size, input_channel):
        self.size = size
        self.input_channel = input_channel

    def decode_image(self, tf_image):
        tf_image = tf.io.decode_raw(tf_image, tf.float32)
        tf_image = tf.reshape(tf_image, (self.size, self.size, self.input_channel))
        return tf_image

    def read_tfrecord(self, example):
        tfrecord_format = (
            {
                'Index': tf.io.FixedLenFeature((), tf.int64),
                'requirements': tf.io.FixedLenFeature((), tf.string),
                't_shape': tf.io.FixedLenFeature((), tf.float32),
                't_location': tf.io.FixedLenFeature((), tf.float32),
                't_scale': tf.io.FixedLenFeature((), tf.float32),
            }
        )
        example = tf.io.parse_single_example(example, tfrecord_format)
        require_input = self.decode_image(example['requirements'])
        return example['Index'], example['t_shape'], example['t_location'], example['t_scale'], require_input

    def get_dataset_next(self, filenames, batch_size):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(100).repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def get_test_dataset(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


def main():
    pressure_path = "F:/SloshingProject/data/osj/train_dsme.csv"
    scaling_pressure_path = "F:/SloshingProject/data/osj/train_1.csv"
    requirements_path = "F:/SloshingProject/data/osj/condition.csv"
    csv_save_path = "F:/SloshingProject/data/osj/requirements.csv"
    tf_save_path = "F:/SloshingProject/data/osj/train_parameters_64.record"

    # generate csv
    g_csv = GenerateCSV(pressure_path, scaling_pressure_path, requirements_path, csv_save_path)
    g_csv.merge_data()

    # generate tfrecord
    g_tf = GenerateTfrecord(csv_save_path, tf_save_path, 64)
    g_tf.create_tfrecord()

    # check_training data
    # l_tf = LoadTfrecord(256, 8)
    # test = l_tf.get_dataset_next(tf_save_path, 16)
    # test_iter = iter(test)
    # for i in range(1000):
    #     idx, shape, loc, scale, require_matrix = next(test_iter)
    #     print(idx, require_matrix)


if __name__ == "__main__":
    main()

