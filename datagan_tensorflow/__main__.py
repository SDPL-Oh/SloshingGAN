HParams = {
    'latent': 16,
    'samples': 3900000,
    'batch': 512,
    'epochs': 100,
    'generator_lr': 0.0003,
    'discriminator_lr': 0.0003,
    'decay_steps': 20000,
    'decay_rate': 0.96,
    'num_gen_data': 500,
    'model_path': 'F:/SloshingProject/weight/',
    'logs_path': 'F:/SloshingProject/logs/',
    'train_data': 'F:/SloshingProject/preprocessing/train_dsme.record',
    'test_data': 'F:/SloshingProject/preprocessing/train_sample.record',
    'input_columns': ['hs', 'tz', 'heading', 'loc'],
    'target_columns': ['pressure']
}

data_dir = 'G:/주제별 자료/02. 데이터 및 프로그램/슬로싱 데이터/20220412_DSME/Data/'
condi_file = 'F:/SloshingProject/preprocessing/sensor.csv'
csv_file = 'F:/SloshingProject/preprocessing/train_dsme.csv'
save_dir = 'F:/SloshingProject/preprocessing/'


def main():
    ################## CSV 생성 ###################
    # from data import GenerateTfrecord, GenerateCSV
    # dsme_data = GenerateCSV(data_dir, condi_file, csv_file)
    # dsme_data.save_csv()

    ################## Tfrecord 생성 ###################
    # sloshing_data = GenerateTfrecord(csv_file, save_dir)
    # sloshing_data.create_tfrecord('train_sample')

    ################### 학습1 ###################
    # from train import SloshingGan
    # sloshing_gan = SloshingGan(HParams)
    # sloshing_gan.train(profiler=False)
    # sloshing_gan.test(csv_file, conditions)

    ################### 학습2 ###################
    from train_2 import Algorithm
    sloshing_gan = Algorithm(HParams)
    sloshing_gan.train()

    # conditions = {
    #     'hs': [10.9],
    #     'tz': [7.5],
    #     'speed': [5],
    #     'heading': [165],
    #     'loc': [6]
    # }
    conditions = 'F:/SloshingProject/data/kbg/condition.csv'
    # sloshing_gan.test(csv_file, conditions)

if __name__ == '__main__':
    main()