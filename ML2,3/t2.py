from t1 import *
from scipy.optimize import brentq

def threshFunc(threshold, data, ESnum, count):
	return fp(data, ESnum, threshold) / float(count) - 0.2

data = read_csv("vyygrusska.csv", ";")

counts = Series.value_counts(data.CLASS)

for i in range(1,4):
	ans = brentq(threshFunc, 0.0, 1.0, args = (data, i, counts.ix[0]))
	print "p" + str(i) + "_Fraud", ans, fp(data, i, ans) / float(counts.ix[0])