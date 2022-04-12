import os
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class PeakData:
    def __init__(self, data_dir, cond_path, save_path, data_header):
        self.data_dir = data_dir
        self.cond_path = cond_path
        self.save_path = save_path
        self.data_header = data_header

    def condition_dat(self):
        return pd.read_csv(self.cond_path)

    def extract_num(self, filename, mode='dat'):
        if mode == 'case':
            return filename[-8:]
        if mode == 'dat':
            return filename[filename.find('ID')+2:filename.find('.dat')]
        else:
            return None

    def read_dat(self):
        nan_list = []
        concat_dat = pd.DataFrame()
        for case in tqdm(os.listdir(self.data_dir), desc='Concatenate case'):
            for dat in os.listdir(os.path.join(self.data_dir, case)):
                data = os.path.join(self.data_dir, case, dat)
                data = pd.read_csv(data, sep="\t", skiprows=1)
                if data.isnull().values.any():
                    nan_list.append((case, '/', dat))
                data = data.dropna(axis=0)
                data.columns = self.data_header
                data['No'] = self.extract_num(case, 'case')
                data['sensor'] = self.extract_num(dat, 'dat')
                concat_dat = pd.concat([concat_dat, data], ignore_index=True)
        print(nan_list)

        return concat_dat

    def concat_cond(self):
        dat_list = self.read_dat()
        cond_list = self.condition_dat()
        dat_list['No'] = dat_list['No'].astype(int)
        cond_list['No'] = cond_list['No'].astype(int)
        data_list = pd.merge(dat_list, cond_list, on='No')
        data_list.to_csv(self.save_path)
        return print('Finished export "{}" file'.format(self.save_path))


class PeckDataset(Dataset):
    def __init__(self, csv_file, cond_header, output_header, transform=None):
        self.cond_header = cond_header
        self.output_header = output_header
        self.peak_data = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.peak_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        conditions_list = self.peak_data[self.cond_header]
        results_list = self.peak_data[self.output_header]
        conditions = conditions_list.iloc[idx]
        results = results_list.iloc[idx]
        sample = {'conditions': conditions, 'results': results}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalized(object):
    def __init__(self, csv_file, cond_header):
        self.peak_data = csv_file
        self.cond_header = cond_header

    def __call__(self, sample):
        conditions = sample['conditions']
        conditions = self.norm(conditions)[self.cond_header].fillna(0)
        return {'conditions': conditions,
                'results': sample['results']}

    def norm(self, x):
        data_stats = self.peak_data.describe()
        data_stats = data_stats.transpose()
        return (x - data_stats['min']) / ((data_stats['max']) - data_stats['min'])


class ToTensor(object):
    def __call__(self, sample):
        conditions, results = sample['conditions'], sample['results']
        return {'conditions': torch.tensor(conditions.values, dtype=torch.float),
                'results': torch.tensor(results.values, dtype=torch.float)}