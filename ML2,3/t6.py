from t1 import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pandas import DataFrame

def vote(ES):
	out = []

	for row in ES:
		p = 0.0
		for pi_Fraud in row:
			p += pi_Fraud

		out.append(p / 3.0)


	return out
			

data = read_csv("vyygrusska.csv", ";")
without_U = data[logical_and(data["CLASS"] != "U", True)].reset_index(drop = True)

y_true = logical_and(without_U["CLASS"] == "F", True)
y_rate = vote(zip(without_U["p1_Fraud"], without_U["p2_Fraud"], without_U["p3_Fraud"]))

FPR, TPR, threshold = roc_curve(y_true, y_rate)
AUC = auc(FPR, TPR)

plt.plot(FPR, TPR, label='%s ROC, AUC = %0.2f, Gini = %0.2f' % ("t6 ansamble", AUC, (AUC * 2) - 1))

plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right', fontsize='small')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()