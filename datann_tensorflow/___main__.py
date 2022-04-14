HParams = {
    'input_size': 5,
    'output_size': 3,
    'batch_size': 512,
    'epochs': 100,
    'lr': 0.0003,
    'model_path': 'F:/SloshingProject/weight/',
    'logs_path': 'F:/SloshingProject/logs/',
    'train_data': 'F:/SloshingProject/preprocessing/train_nn.record',
    'test_data': 'F:/SloshingProject/preprocessing/train_nn.record',
    'input_columns': ['hs', 'tz', 'speed', 'heading', 'sensor'],
    'target_columns': ['shape', 'location', 'scale']
}

data_dir = 'G:/주제별 자료/02. 데이터 및 프로그램/슬로싱 데이터/20220412_DSME/Data/'
condi_file = 'F:/SloshingProject/preprocessing/sensor.csv'
csv_file = 'F:/SloshingProject/preprocessing/train_nn.csv'
save_dir = 'F:/SloshingProject/preprocessing/'

conditions = {
    'hs': [5.9],
    'tz': [6.5],
    'speed': [5],
    'heading': [120],
    'sensor': [1]
}

def main():
    ################## DATA 생성 ###################
    # from data import GenerateCSV, GenerateTfrecord
    # csv = GenerateCSV(data_dir, condi_file, csv_file)
    # tf = GenerateTfrecord(csv_file, save_dir)
    # csv.save_csv()
    # tf.create_tfrecord('train_nn')

    ################### 학습 ###################
    from train import Algorithm
    sloshing_nn = Algorithm(HParams)
    sloshing_nn.train()
    # sloshing_nn.test(csv_file, conditions)

if __name__ == '__main__':
    main()