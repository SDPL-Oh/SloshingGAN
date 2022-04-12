import os
from data import GenerateTfrecord
from train import SloshingGan

here = os.path.dirname(os.path.abspath(__file__))

HParams = {
    'input_size': 5,
    'output_size': 1,
    'latent': 16,
    'samples': 600000,
    'batch': 512,
    'epochs': 100,
    'generator_lr': 0.0001,
    'discriminator_lr': 0.0001,
    'decay_steps': 20000,
    'decay_rate': 0.96,
    'num_gen_data': 100,
    'model_path': 'F:/SloshingProject/weight/',
    'logs_path': 'F:/SloshingProject/logs/',
    'train_data': 'F:/SloshingProject/preprocessing/train.record',
    'test_data': 'F:/SloshingProject/preprocessing/train.record',
    'input_columns': ['Hs', 'Tz', 'Speed', 'Heading', 'sensor'],
    'target_columns': ['Peak Pressure(bar)']
}

def main():
    ################## Tfrecord 생성 명령 ###################
    csv_file = 'F:/SloshingProject/preprocessing/train.csv'
    save_dir = 'F:/SloshingProject/preprocessing/'
    # sloshing_data = GenerateTfrecord(csv_file, save_dir)
    # sloshing_data.create_tfrecord('train')

    ################### 학습/평가 명령 ###################
    sloshing_gan = SloshingGan(HParams)
    # sloshing_gan.train(profiler=False)

    conditions = {
        'Hs': [9.8],
        'Tz': [8.5],
        'Speed': [5],
        'Heading': [90],
        'sensor': [1]
    }
    sloshing_gan.test(csv_file, conditions)

if __name__ == '__main__':
    main()