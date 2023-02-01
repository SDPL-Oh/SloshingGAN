import pandas as pd
import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt


def estimate_para_scipy(pressures):
    # The default is “MLE” (Maximum Likelihood Estimate); “MM” (Method of Moments) is also available.
    shape, loc, scale = stats.weibull_min.fit(pressures, method="MM")
    return shape, scale, loc


def gamma(x):
    # return tf.math.lgamma(x).numpy() is the lower incomplete Gamma function.
    # return tf.math.igammac(x).numpy() is the upper incomplete Gamma function.
    # return tf.math.lgamma(x).numpy() this function computes log((input - 1)!) for every element in the tensor.
    return special.gamma(x)


def skew_equation(shape):
    skew = (gamma(1 + (3 / shape)) - 3 * gamma(1 + (1 / shape)) *
            gamma(1 + (2 / shape)) + 2 * gamma(1 + (1 / shape)) ** 3) / \
           (gamma(1 + (2 / shape)) - gamma(1 + (1 / shape)) ** 2) ** (3 / 2)
    return skew


def var_equation(shape, scale):
    var = (scale ** 2) * (gamma(1 + (2 / shape)) - gamma(1 + (1 / shape)) ** 2)
    return var


def mean_equation(shape, scale, location):
    mean = scale * gamma(1 + (1 / shape)) + location
    return mean


def estimate_para_cal(shape, scale, location):
    cdf_skew = skew_equation(shape)
    cdf_var = var_equation(shape, scale)
    cdf_mean = mean_equation(shape, scale, location)
    return cdf_skew, cdf_var, cdf_mean


def weibull_random(num_data, shape, scale, location):
    return stats.weibull_min.rvs(shape, loc=location, scale=scale, size=num_data, random_state=None)


def plot_weibull(data_1, data_2, save_path):
    shape_1, loc_1, scale_1 = estimate_para_scipy(data_1)
    data_1 = data_1[data_1 > scale_1]
    plt.figure(figsize=(6.5, 6))
    plt.scatter(data_1,
                1 - stats.weibull_min.cdf(x=data_1, c=shape_1, loc=loc_1, scale=scale_1),
                marker='+',
                label="predict",
                color='r',
                linewidth=1
                )

    shape_2, loc_2, scale_2 = estimate_para_scipy(data_2)
    data_2 = data_2[data_2 > scale_2]
    plt.scatter(data_2,
                1 - stats.weibull_min.cdf(x=data_2, c=shape_2, loc=loc_2, scale=scale_2),
                marker='+',
                label="true",
                color="k",
                linewidth=1
                )

    plt.yscale('log')
    plt.xlabel('Pressure')
    plt.ylabel('Probability of exceedance')
    plt.axis([0, 0.5, 0.00001, 1])
    plt.legend()
    plt.savefig(save_path)
    plt.clf()


def normal_distribution(data, parameter, save_path):
    label_list = ['p_shape', 'p_location', 'p_scale',
                  'r_shape', 'r_location', 'r_scale',
                  's_shape', 's_location', 's_scale']
    img_size = {'shape': 1.2, 'location': 0.035, 'scale': 0.06}

    data_list = [label for label in label_list if '{}'.format(parameter) in label]
    plt.figure(figsize=(6.5, 6))
    plt.xlim([0, img_size[parameter]])
    data_stats = data.describe()
    for label, color in zip(data_list, ['red', 'black', 'blue']):
        plt.scatter(
            data[label],
            stats.norm.pdf(
                data[label],
                data_stats[label]['mean'],
                data_stats[label]['std']),
            label=label,
            c=color,
            s=3
        )
        plt.hist(
            data[label],
            color=color,
            bins=25,
            density=True,
            alpha=0.6)

    plt.title(parameter)
    plt.xlabel('x')
    plt.ylabel(parameter)
    plt.legend()
    plt.savefig(save_path)
    plt.show()


class EstimateParameters:
    def __init__(self, pressure_path, requirements_path, save_path):
        self.pressure_path = pressure_path
        self.requirements_path = requirements_path
        self.save_path = save_path
        self.input_columns = ['hs', 'tz', 'speed', 'heading', 'loc']

    def data_estimate(self):
        pressures = pd.read_csv(self.pressure_path)
        requirements = pd.read_csv(self.requirements_path)

        for requirement in requirements.itertuples():
            raw_data = self.extract_data(
                pressures,
                requirements.iloc[requirement.Index:requirement.Index + 1].reset_index()
            )

            if raw_data is not None:
                shape, scale, location = estimate_para_scipy(raw_data)
                data_mean = np.mean(raw_data)
                data_var = np.var(raw_data)
                data_skew = stats.skew(raw_data)
                cdf_skew, cdf_var, cdf_mean = estimate_para_cal(shape, scale, location)

                random_pressures = weibull_random(65536, shape, scale, location)
                random_mean = np.mean(random_pressures)
                random_var = np.var(random_pressures)
                random_skew = stats.skew(random_pressures)

                print("### {} {} Compared parameter ###".format(requirement.Index, len(raw_data)))
                print("         data    cdf     random")
                print("Mean     {:.5f} {:.5f} {:.5f}".format(data_mean, cdf_mean, random_mean))
                print("Variance {:.5f} {:.5f} {:.5f}".format(data_var, cdf_var, random_var))
                print("Skewness {:.5f} {:.5f} {:.5f}".format(data_skew, cdf_skew, random_skew))

                plot_weibull(random_pressures, raw_data, self.save_path + "{}.jpg".format(requirement.Index))
            else:
                pass

    def extract_data(self, df, requirements):
        for col in self.input_columns:
            required = (df[col] == requirements[col][0])
            df = df[required]
        if df.empty:
            return None
        else:
            return df['pressure']


def main():
    pressure_path = "F:/SloshingProject/data/kbg/paper/train_dsme.csv"
    requirements_path = "F:/SloshingProject/data/kbg/paper/condition.csv"
    save_path = "F:/SloshingProject/data/kbg/asset/"

    analysis = EstimateParameters(pressure_path, requirements_path, save_path)
    analysis.data_estimate()


if __name__ == "__main__":
    main()


