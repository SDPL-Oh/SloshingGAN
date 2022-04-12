import os
from data import GenerateTfrecord, GenerateCSV
from train import SloshingGan

here = os.path.dirname(os.path.abspath(__file__))

HParams = {
    'input_size': 5,
    'output_size': 1,
    'latent': 16,
    'samples': 3900000,
    'batch': 512,
    'epochs': 100,
    'generator_lr': 0.0001,
    'discriminator_lr': 0.0001,
    'decay_steps': 20000,
    'decay_rate': 0.96,
    'num_gen_data': 100,
    'model_path': 'F:/SloshingProject/weight/',
    'logs_path': 'F:/SloshingProject/logs/',
    'train_data': 'F:/SloshingProject/preprocessing/train_dsme.record',
    'test_data': 'F:/SloshingProject/preprocessing/train_dsme.record',
    'input_columns': ['hs', 'tz', 'speed', 'heading', 'loc'],
    'target_columns': ['pressure']
}

data_dir = 'G:/주제별 자료/02. 데이터 및 프로그램/슬로싱 데이터/20220412_DSME/Data/'
condi_file = 'F:/SloshingProject/preprocessing/sensor.csv'
csv_file = 'F:/SloshingProject/preprocessing/train_dsme.csv'
save_dir = 'F:/SloshingProject/preprocessing/'

conditions = {
    'hs': [9.8],
    'tz': [8.5],
    'speed': [5],
    'heading': [90],
    'loc': [1]
}

def main():
    ################## CSV 생성 ###################
    # dsem_data = GenerateCSV(data_dir, condi_file, csv_file)
    # dsem_data.save_csv()

    ################## Tfrecord 생성 ###################
    # sloshing_data = GenerateTfrecord(csv_file, save_dir)
    # sloshing_data.create_tfrecord('train_dsme')

    ################### 학습 ###################
    sloshing_gan = SloshingGan(HParams)
    sloshing_gan.train(profiler=False)

    ################### 생성 ###################
    # sloshing_gan.test(csv_file, conditions)

if __name__ == '__main__':
    main()