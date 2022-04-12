import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

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
            'Hs': self.float_feature(data.Hs),
            'Tz': self.float_feature(data.Tz),
            'Speed': self.float_feature(data.Speed),
            'Heading': self.float_feature(data.Heading),
            'Sensor': self.float_feature(data.sensor),
            'Pressure': self.float_feature(data._2),
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=label_dict))
        return tf_example

    def create_tfrecord(self, mode):
        writer = tf.io.TFRecordWriter(os.path.join(self.save_dir + "{}.record".format(mode)))
        data_info = pd.read_csv(self.csv_file)
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
