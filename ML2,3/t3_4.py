from t1 import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pandas import DataFrame

data = read_csv("vyygrusska.csv", ";")


for i in range(1,4):

	y_true = logical_and(data[data["CLASS"] != 'U']["CLASS"] == "F", True).reset_index(drop = True)
	y_rate = data[data["CLASS"] != "U"]['p' + str(i) + '_Fraud'].reset_index(drop = True)
	print y_true

	FPR, TPR, threshold = roc_curve(y_true, y_rate)
	AUC = auc(FPR, TPR)

	plt.plot(FPR, TPR, label='%s ROC, AUC = %0.2f, Gini = %0.2f' % ('p' + str(i) + '_Fraud', AUC, (AUC * 2) - 1))


plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right', fontsize='small')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()