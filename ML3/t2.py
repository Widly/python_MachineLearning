import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.optimize import curve_fit

def penalty_func(x, A, sigma):
	#return A - (np.exp(-(x)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)))
	return A - np.exp(-(x**2/(2.0*sigma**2)))/sigma

def polynom(x, a, b, c, d, e):
	return a*x**4 + b*x**3 + c*x**2 + d*x + e

def calculate_penalty(data, A, sigma, coeffs):
	penalty_array = []
	for row in data:
		delta = row[1] - polynom(row[0], coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4])
		penalty_array.append(penalty_func(delta, A, sigma))

	return penalty_array

data = read_csv('regression_x_y.2.csv', ';')


data = np.array(data)

y_max = max(data[:, 1])
for i in range(0, len(data)):
	data[i, 1] = data[i, 1] / 300

coeffs, covar = curve_fit(polynom, data[:, 0], data[:, 1])
print coeffs
print sum(calculate_penalty(data, 1.0, 1.0, coeffs))