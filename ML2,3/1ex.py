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

h = .02  # размер шага в зацеплении(mesh)
max_row = 8000
border = 4000

# загрузка csv и удаление неопознанных строк
data = read_csv("vyygrusska.csv", ";")
data = data[np.logical_and(data["CLASS"] != "U", True)].reset_index(drop = True)

# выбор признаков, по которым будет происходить классификация
X = data[["X13", "X17"]].iloc[0 : max_row].astype(np.float64)
y = np.logical_and(data["CLASS"] == "F", True).iloc[0 : max_row]

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

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
         "Quadratic Discriminant Analysis"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability = True),
    SVC(gamma=2, C=1, probability = True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

for nm, clf in zip(names, classifiers):

	clf.fit(X_train, y_train)
	predict = clf.predict_proba(X_test)[:, 1]

	FPR, TPR, threshold = roc_curve(y_test, predict)
	AUC = auc(FPR, TPR)

	plt.plot(FPR, TPR, label='%s ROC, AUC = %0.2f, Gini = %0.2f' % (nm, AUC, (AUC * 2) - 1), linewidth = 1.5)


plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right', fontsize='medium')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()