import os
import math
import numpy as np
from scipy.stats import exponweib, weibull_min
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
                    exponweib.pdf(data_list, *exponweib.fit(data_list, 1, 1, scale=2, loc=0)),
                    label='gen')
        plt.scatter(data_info[self.target_columns][:62],
                    exponweib.pdf(data_info[self.target_columns][:62],
                                  *exponweib.fit(data_info[self.target_columns][:62],
                                                 1, 1, scale=2, loc=0)),
                    label='exp')

        plt.ylim([0, 5])
        plt.legend()
        plt.show()

    def plot_weibull_scipy(self, data, raw_data, conditions, save_path):
        p_shape, p_loc, p_scale = weibull_min.fit(data, method="MM")
        data = data[data > p_scale]
        plt.scatter(data,
                    1-weibull_min.cdf(x=data, c=p_shape, loc=p_loc, scale=p_scale),
                    color='r',
                    label='gen')

        r_shape, r_loc, r_scale = weibull_min.fit(raw_data, method="MM")
        raw_data = raw_data[raw_data > r_scale]
        plt.scatter(raw_data,
                    1-weibull_min.cdf(x=raw_data, c=r_shape, loc=r_loc, scale=r_scale),
                    color='b',
                    label='exp')

        plt.title(conditions.values[0])
        plt.yscale('log')
        plt.xlabel('Pressure')
        plt.ylabel('Probability of exceedance')
        plt.axis([0, 0.5, 0.00001, 1])
        plt.legend()
        # plt.show()
        plt.savefig(save_path + '.jpg')
        plt.clf()
        return [p_shape, p_loc, p_scale, r_shape, r_loc, r_scale]

    def parameters(self, data):
        peak_press_lst = data
        num = len(peak_press_lst)

        sample_mean = sum(peak_press_lst) / num
        sample_var = 0
        for i in range(num):
            sample_var += (peak_press_lst[i] - sample_mean) ** 2
        sample_var = sample_var / (num - 1)
        sample_std = math.sqrt(sample_var)
        sample_skew = 0
        for i in range(num):
            sample_skew += ((peak_press_lst[i] - sample_mean) / sample_std) ** 3
        sample_skew = sample_skew / num

        gam = 3.5
        gam_lst = np.linspace(gam, 0, 10000, endpoint=False)
        for i in range(len(gam_lst)):
            gam = gam_lst[i]
            err = abs(self.gamma_fitting(gam) - sample_skew) / sample_skew
            if err < 0.005:
                break
            else:
                continue

        beta = math.sqrt(sample_var / (math.gamma(1 + 2 / gam) - (math.gamma(1 + 1 / gam) ** 2)))
        delta = sample_mean - beta * math.gamma(1 + 1 / gam)

        gam = round(gam, 6)
        beta = round(beta, 6)
        delta = round(delta, 6)

        return gam, beta, delta

    def gamma_fitting(self, gam):
        return (math.gamma(1 + 3 / gam) - 3 * math.gamma(1 + 1 / gam) * math.gamma(1 + 2 / gam) + 2 * (
            math.gamma(1 + 1 / gam)) ** 3) \
               / ((math.gamma(1 + 2 / gam) - (math.gamma(1 + 1 / gam)) ** 2) ** (3 / 2))

    def weibull_cdf(self, x, gam, beta, delta):
        return 1 - math.exp(-(((x - delta) / beta) ** gam))

    def exceedance_probability(self, x, gam, beta, delta):
        return 1 - self.weibull_cdf(x, gam, beta, delta)

    def plot_weibull_math(self, data):
        gam, beta, delta = self.parameters(data)
        prob_lst = []
        for i in range(len(data)):
            epf = self.exceedance_probability(data[i], gam, beta, delta)
            prob_lst.append(epf)

        plt.scatter(data, prob_lst, color='b', marker='+',  label='1')

        plt.axis([0, 0.5, 0.00001, 1])
        plt.yscale('log')
        plt.legend()
        plt.show()

    def extract_dat(self, df, conditions, columns):
        for col in columns:
            condi = (df[col] == conditions[col][0])
            df = df[condi]
        return df['pressure']


