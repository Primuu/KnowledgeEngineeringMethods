import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy


def mean(values):
    if len(values) == 0:
        return 0
    return sum(values) / len(values)


def sample_standard_deviation(values, mean_value):
    if len(values) == 0:
        return 0
    squares_sum = 0
    for value in values:
        squares_sum += (value - mean_value) ** 2
    return np.sqrt(squares_sum / (len(values) - 1))


def pearson_correlation_coefficient(x_sum, y_sum, n, xy_sum, x2_sum, y2_sum):
    return (n * xy_sum - x_sum * y_sum) / np.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))


# Creating simple Data Frame
df = pd.DataFrame()
df['X'] = [1, 2, 3, 4, 5]
df['Y'] = [4, 6, 9, 11, 18]
print("Data Frame:")
print(df)

# Creating chart
plt.scatter(df['X'], df['Y'], label='Independent Values')
plt.xlabel('Values X')
plt.ylabel('Values Y')
plt.legend()
plt.show()

# Counting mean values
np_mean_x = np.mean(df['X'])
np_mean_y = np.mean(df['Y'])

mean_x = mean(df['X'])
mean_y = mean(df['Y'])

print("\nMean X counted using my function: ", mean_x)
print("Mean X counted using numpy function: ", np_mean_x)
print("\nMean Y counted using my function: ", mean_y)
print("Mean Y counted using numpy function: ", np_mean_y)

# Counting sample standard deviation
np_sx = np.std(df['X'])
np_sy = np.std(df['Y'])

sx = sample_standard_deviation(df['X'], mean_x)
sy = sample_standard_deviation(df['Y'], mean_y)

print("\nStandard deviation X counted using my function: ", sx)
print("Standard deviation X counted using numpy function: ", np_sx)
print("\nStandard deviation Y counted using my function: ", sy)
print("Standard deviation Y counted using numpy function: ", np_sy)

# Counting Pearson's correlation coefficient
n = len(df['X'])

pearson = pd.DataFrame(df[:])  # creating new data frame based on first df
pearson['y2'] = df['Y'] * df['Y']
pearson['xy'] = df['X'] * df['Y']
pearson['x2'] = df['X'] * df['X']
pearson.loc['sum'] = pearson.sum()  # add row containing sum of each column

print("\nn =", n)
print()
print(pearson)

pearson_scipy = scipy.stats.pearsonr(df['X'], df['Y'])

pearson_r = pearson_correlation_coefficient(pearson.loc['sum', 'X'], pearson.loc['sum', 'Y'],
                                            n, pearson.loc['sum', 'xy'], pearson.loc['sum', 'x2'],
                                            pearson.loc['sum', 'y2'])

print("\nPearson's correlation coefficient counted using my function: ", pearson_r)
print("Pearson's correlation coefficient counted using numpy function: ", pearson_scipy[0])

# Summary of the obtained results
b = pearson_r * sy / sx
a = mean_y - b * mean_x

print("\nValues differ from those given in the PDF because they are not rounded.")
print("b: ", b)
print("a: ", a)

# Calculation of the best-fit line


def regression_line(b, x, a):
    return (b * x) + a


# Creating chart with best-fit line
x = np.linspace(0, 5, 1000)
plt.scatter(df['X'], df['Y'], label="Independent Values")
plt.plot(x, regression_line(b, x, a), 'r', label="Regression Line")
plt.xlabel('Values X')
plt.ylabel('Values Y')
plt.legend()
plt.show()

# Predicting next values
df = df._append({'X': 6, 'Y': np.nan}, ignore_index=True)
print("\nData Frame with new value:")
print(df)

df.at[5, 'Y'] = regression_line(b, df['X'][5], a)
print("\nData Frame with new predicted value:")
print(df)

# Predicting values for X=7 and X=8
df = df._append({'X': 7, 'Y': np.nan}, ignore_index=True)
df = df._append({'X': 8, 'Y': np.nan}, ignore_index=True)
print("\nData Frame with new values 7 and 8:")
print(df)

df.at[6, 'Y'] = regression_line(b, df['X'][6], a)
df.at[7, 'Y'] = regression_line(b, df['X'][7], a)
print("\nData Frame with new predicted value for X=7 and X=8:")
print(df)
