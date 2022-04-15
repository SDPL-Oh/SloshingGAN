import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf


class GenerateCSV:
    def __init__(self, data_dir, cond_path, save_path):
        self.data_dir = data_dir
        self.cond_path = cond_path
        self.save_path = save_path
        self.data_header = ['pressure', 'time', 'rise', 'duration']

    def concat_dat(self, filename):
        concat_dat = pd.DataFrame()
        exp = np.int16(filename[filename.find('RNO') + 3:filename.find('_0')])
        filling = np.float32(filename[filename.find('F') + 1:filename.find('H')]) / 100
        hs = np.float32(filename[filename.find('H') + 1:filename.find('T')]) / 10
        tz = np.float32(filename[filename.find('T') + 1:filename.find('HD')]) / 10
        heading = np.float32(filename[filename.find('HD') + 2:filename.find('V')])
        speed = np.float32(filename[filename.find('V') + 1:filename.find('RN')]) / 10
        locations = pd.read_csv(self.cond_path)
        condi_path = os.path.join(self.data_dir, filename)
        dat_list = os.listdir(condi_path)
        for dat in dat_list:
            dat_path = os.path.join(condi_path, dat)
            sensor = int(pd.read_csv(dat_path, sep="\t").columns[1])
            dat = pd.read_csv(dat_path, sep="\t", skiprows=1)
            dat.columns = self.data_header
            dat['sensor'] = sensor
            dat['exp'] = exp
            dat['filling'] = filling
            dat['hs'] = hs
            dat['tz'] = tz
            dat['heading'] = heading
            dat['speed'] = speed
            dat = pd.merge(dat, locations, on='sensor')
            concat_dat = pd.concat([concat_dat, dat], ignore_index=True)
        return concat_dat

    def save_csv(self):
        condi_list = os.listdir(self.data_dir)
        concat_dat = pd.DataFrame()
        for condi in tqdm(condi_list, desc='Generate train csv file'):
            dat = self.concat_dat(condi)
            concat_dat = pd.concat([concat_dat, dat], ignore_index=True)
        concat_dat.to_csv(self.save_path)
        print('Finished export "{}" file'.format(self.save_path))


class GenerateTfrecord:
    def __init__(self, csv_file, save_dir):
        self.csv_file = csv_file
        self.save_dir = save_dir

    def bytes_feature(self, values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    def float_feature(self, values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def int64_feature(self, values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def create_tfvalue(self, data):
        label_dict = {
            'Index': self.float_feature(data.Index),
            'hs': self.float_feature(data.hs),
            'tz': self.float_feature(data.tz),
            'speed': self.float_feature(data.speed),
            'heading': self.float_feature(data.heading),
            'sensor': self.float_feature(data.sensor),
            'loc': self.float_feature(data.loc),
            'pressure': self.float_feature(data.pressure),
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=label_dict))
        return tf_example

    def create_tfrecord(self, mode):
        writer = tf.io.TFRecordWriter(os.path.join(self.save_dir + "{}.record".format(mode)))
        data_info = pd.read_csv(self.csv_file)[:3000]
        data_info = self.normalized(data_info, data_info).fillna(0)
        for data in tqdm(data_info.itertuples(), desc='generate tfrecord values', total=len(data_info)):
            tf_example = self.create_tfvalue(data)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecord(value)')

    def normalized(self, data, standard_data):
        data_stats = standard_data.describe()
        data_stats = data_stats.transpose()
        trans_data = (data - data_stats['min']) / ((data_stats['max']) - data_stats['min'])
        return trans_data.replace([-np.inf, np.inf], 0)


class LoadTfrecord:
    def __init__(self, epochs):
        self.epochs = epochs

    def read_tfrecord(self, example):
        tfrecord_format = ({
            'Index': tf.io.FixedLenFeature((), tf.float32),
            'hs': tf.io.VarLenFeature(tf.float32),
            'tz': tf.io.VarLenFeature(tf.float32),
            'speed': tf.io.VarLenFeature(tf.float32),
            'heading': tf.io.VarLenFeature(tf.float32),
            'sensor': tf.io.VarLenFeature(tf.float32),
            'loc': tf.io.VarLenFeature(tf.float32),
            'pressure': tf.io.VarLenFeature(tf.float32)
        })
        example = tf.io.parse_single_example(example, tfrecord_format)
        inputs = tf.sparse.concat(axis=0, sp_inputs=[
            example['hs'],
            example['tz'],
            example['speed'],
            example['heading'],
            example['loc']])
        inputs = tf.sparse.to_dense(inputs)
        return example['Index'], inputs, example['pressure']

    def load_data(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.read_tfrecord,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def get_dataset(self, filenames, batch_size, samples, is_training=True):
        dataset = self.load_data(filenames)
        if is_training:
            dataset = dataset.shuffle(samples, reshuffle_each_iteration=False)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).repeat(self.epochs)
        return iter(dataset)

    def get_dataset_next(self, filenames, batch_size):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.read_tfrecord,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset