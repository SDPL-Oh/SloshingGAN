from scipy import stats
import matplotlib.pyplot as plt

class WeibullDistribution:
    def __init__(self, target_columns):
        self.target_columns = target_columns

    def normalized(self, data, standard_data, columns):
        data_stats = standard_data.describe()
        data_stats = data_stats.transpose()
        trans_data = (data - data_stats['min']) / ((data_stats['max']) - data_stats['min'])
        trans_data = trans_data[columns].fillna(0)
        return trans_data.astype('float32')

    def denormalized(self, data, standard_data, columns):
        data_stats = standard_data.describe()
        data_stats = data_stats.transpose()
        trans_data = (data * ((data_stats['max'][columns]) - data_stats['min'][columns])) + data_stats['min'][columns]
        return trans_data

    def plot_scatter(self, data_list, data_info):
        plt.scatter(data_list,
                    stats.exponweib.pdf(data_list, *stats.exponweib.fit(data_list, 1, 1, scale=2, loc=0)),
                    label='gen')
        plt.scatter(data_info[self.target_columns][:62],
                    stats.exponweib.pdf(data_info[self.target_columns][:62],
                                        *stats.exponweib.fit(data_info[self.target_columns][:62],
                                                             1, 1, scale=2, loc=0)),
                    label='exp')
        plt.legend()
        plt.show()