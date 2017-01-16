import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def penalty_func(x, A, sigma):
	#return np.exp(-(x**2/(2.0*sigma**2)))/(sigma*np.sqrt(2.0*np.pi))
	return np.exp(-(x**2/(2.0*sigma**2)))/sigma

penalty_func(6.5, 1.0, 10.0)

X = np.arange(-10, 10, 0.1)
Y = []

for row in X:
	Y.append(penalty_func(row, 1.0, 0.1))


plt.figure()
plt.plot(X, Y, linewidth = 1.5)

#plt.legend(loc='lower right', fontsize='medium')
plt.show()