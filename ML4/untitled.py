# coding: utf-8

import pandas as pn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

data = pn.read_csv('dataset.txt', sep = '\t')

learning_count = 200
experiment_count = 2000

# выборка
X = data[["0", "2", "21"]].ix[0 : experiment_count].astype(np.float64)
y = np.logical_and(data["17"] == -1, True).ix[0 : experiment_count]


X = np.array(X)
y = np.array(y)

# стандартизация
X = StandardScaler().fit_transform(X)

# обучающая выборка
X_train = X[0 : learning_count - 1]
y_train = y[0 : learning_count - 1]

classifiers = [KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis()]

classifiers_nums = zip(range(0, 3), [0, 1, 2])
weights = [1.0, 1.0, 1.0]

decr = 0.02
incr = decr / 2.0
step = 40

for num, clf in classifiers_nums:
	classifiers[clf].fit(X_train, y_train)
	print classifiers[clf].score(X[1500 : 1700], y[1500 : 1700])

print
for i in range(0, 30):
	for num, clf in classifiers_nums:
		score = classifiers[clf].score(X[learning_count + step * i : learning_count + step * (i + 1) - 1], 
			y[learning_count + step * i : learning_count + step * (i + 1) - 1])
		
		#print score

		'''
		if (score > 0.8):
			weights[num] += (1.0 - weights[num]) * 0.05
		else:
			weights[num] -= weights[num] * 0.01
		'''


		if (score > 0.80):
			weights[num] += incr
		else:
			weights[num] -= decr

		if (weights[num] > 1.0):
			weights[num] = 1.0



print weights[0], weights[1], weights[2]