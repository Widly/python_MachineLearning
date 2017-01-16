import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

def del_blowouts(data, threshold):
	prev = data["Y"].ix[0]
	to_delete = []

	for i, row in enumerate(data['Y']):
		if abs(row - prev) > threshold:
			to_delete.append(False)
			continue

		to_delete.append(True)
		prev = row

	return data[to_delete].reset_index(drop = True), len(to_delete) - sum(to_delete)


data = read_csv("regression_x_y.csv", ";")

fitted_data, deleted_num = del_blowouts(data, 800)
fitted_data.to_csv('regression_x_y.2.csv', index=False, sep = ';')


plt.figure()
plt.plot(fitted_data["X"], fitted_data["Y"], label = '%d deleted points' % deleted_num, linewidth = 1.5)
plt.legend(loc='lower right', fontsize='medium')
plt.show()