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

    def plot_weibull_scipy(self, parameter, raw_data, conditions):
        shape, loc, scale = weibull_min.fit(raw_data, method="MM")
        raw_data = raw_data[raw_data > scale]
        plt.scatter(raw_data,
                    1-weibull_min.cdf(x=raw_data, c=shape, loc=loc, scale=scale),
                    color='b',
                    label='exp')

        linear_data = np.linspace(raw_data.min, raw_data.max)

        plt.scatter(linear_data,
                    1-weibull_min.cdf(x=linear_data, c=parameter[0], loc=parameter[1], scale=parameter[2]),
                    color='b',
                    label='exp')

        plt.title('hz{}, tz{}, sp{}, hd{}, sensor{}'.format(*conditions.values()))
        plt.yscale('log')
        plt.xlabel('Pressure')
        plt.ylabel('Probability of exceedance')
        plt.axis([0, 0.5, 0.00001, 1])
        plt.legend()
        plt.show()

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

    def extract_dat(self, df, conditions):
        condi = (df['hs'] == conditions['hs'][0]) \
                & (df['tz'] == conditions['tz'][0]) \
                & (df['speed'] == conditions['speed'][0]) \
                & (df['heading'] == conditions['heading'][0]) \
                & (df['sensor'] == conditions['sensor'][0])
        return df[condi]['pressure']
