# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc

def vote(ES):
	out = []
	divider = float(len(ES[0]))

	for row in ES:
		p = 0.0
		for pi_Fraud in row:
			p += pi_Fraud

		out.append(p / divider)

	return out


h = .02  # размер шага в зацеплении(mesh)
max_row = 15000
border = 7000

# загрузка csv и удаление неопознанных строк
data = read_csv("vyygrusska.csv", ";")
data = data[np.logical_and(data["CLASS"] != "U", True)].reset_index(drop = True)

# выбор признаков, по которым будет происходить классификация
X = data[["X17", "X18"]].iloc[0 : max_row].astype(np.float64)
y = np.logical_and(data["CLASS"] == "F", True).iloc[0 : max_row]

# оценки классификаторов из csv
X_csv_clfs = np.array(data[["p1_Fraud", "p2_Fraud", "p3_Fraud", "p4_Fraud", "p5_Fraud"]].iloc[border : max_row, :])

# подготовка данных для передачи классификаторам
X = np.array(X)
y = np.array(y)

X = StandardScaler().fit_transform(X) # стандартизация


# обучающая выборка
X_train = X[0 : border - 1, :]
y_train = y[0 : border - 1]

# контрольная выборка
X_test = X[border : max_row, :]
y_test = y[border : max_row]

clf1 = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
clf1.fit(X_train, y_train)
predict1 = clf1.predict_proba(X_test)[:, 1]

clf2 = AdaBoostClassifier()
clf2.fit(X_train, y_train)
predict2 = clf2.predict_proba(X_test)[:, 1]

clf3 = KNeighborsClassifier(3)
clf3.fit(X_train, y_train)
predict3 = clf3.predict_proba(X_test)[:, 1]

ensemble_predict = vote(zip(predict1, predict2))

FPR, TPR, threshold = roc_curve(y_test, ensemble_predict)
AUC = auc(FPR, TPR)

plt.plot(FPR, TPR, label='%s ROC, AUC = %0.2f, Gini = %0.2f' % ("Ensemble", AUC, (AUC * 2) - 1), linewidth = 1.5)


for i in range(1,6):
	FPR, TPR, threshold = roc_curve(y_test, X_csv_clfs[:, i - 1])
	AUC = auc(FPR, TPR)

	plt.plot(FPR, TPR, label='%s ROC, AUC = %0.2f, Gini = %0.2f' % ('p' + str(i) + '_Fraud', AUC, (AUC * 2) - 1))


plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right', fontsize='medium')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()