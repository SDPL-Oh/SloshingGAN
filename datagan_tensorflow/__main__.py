import os
from data import GenerateTfrecord, GenerateCSV
from train import SloshingGan
from train_2 import Algorithm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
here = os.path.dirname(os.path.abspath(__file__))

HParams = {
    'input_size': 5,
    'output_size': 1,
    'latent': 16,
    'samples': 3900000,
    'batch': 512,
    'epochs': 10,
    'generator_lr': 0.0003,
    'discriminator_lr': 0.0003,
    'decay_steps': 20000,
    'decay_rate': 0.96,
    'num_gen_data': 100,
    'model_path': 'F:/SloshingProject/weight/',
    'logs_path': 'F:/SloshingProject/logs/',
    'train_data': 'F:/SloshingProject/preprocessing/train_sensor.record',
    'test_data': 'F:/SloshingProject/preprocessing/train_sensor.record',
    'input_columns': ['hs', 'tz', 'speed', 'heading', 'sensor'],
    'target_columns': ['pressure']
}

data_dir = 'G:/주제별 자료/02. 데이터 및 프로그램/슬로싱 데이터/20220412_DSME/Data/'
condi_file = 'F:/SloshingProject/preprocessing/sensor.csv'
csv_file = 'F:/SloshingProject/preprocessing/train_dsme.csv'
save_dir = 'F:/SloshingProject/preprocessing/'

conditions = {
    'hs': [5.9],
    'tz': [6.5],
    'speed': [5],
    'heading': [120],
    'sensor': [1]
}

def main():
    ################## CSV 생성 ###################
    # dsme_data = GenerateCSV(data_dir, condi_file, csv_file)
    # dsme_data.save_csv()

    ################## Tfrecord 생성 ###################
    # sloshing_data = GenerateTfrecord(csv_file, save_dir)
    # sloshing_data.create_tfrecord('train_temp')

    ################### 학습 ###################
    # sloshing_gan = SloshingGan(HParams)
    # sloshing_gan.train(profiler=False)

    ################### 생성 ###################
    # sloshing_gan.test(csv_file, conditions)

    ################### 학습 ###################
    sloshing_gan = Algorithm(HParams)
    sloshing_gan.train()

if __name__ == '__main__':
    main()