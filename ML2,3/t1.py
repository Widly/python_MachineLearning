from pandas import read_csv, Series
from numpy import logical_and

def tp(data, ESnum, threshold):
	ESname = "p" + str(ESnum) + "_Fraud"
	return len(data[logical_and(data["CLASS"] == 'F', data[ESname] >= threshold)])


def fp(data, ESnum, threshold):
	ESname = "p" + str(ESnum) + "_Fraud"
	return len(data[logical_and(data["CLASS"] == 'G', data[ESname] >= threshold)])


def tn(data, ESnum, threshold):
	ESname = "p" + str(ESnum) + "_Fraud"
	return len(data[logical_and(data["CLASS"] == 'G', data[ESname] <= threshold)])


def fn(data, ESnum, threshold):
	ESname = "p" + str(ESnum) + "_Fraud"
	return len(data[logical_and(data["CLASS"] == 'F', data[ESname] <= threshold)])

if __name__ ==  "__main__":
	threshold = 0.5
	data = read_csv("vyygrusska.csv", ";")

	counts = Series.value_counts(data.CLASS)

	for i in range(1,4):
		print "\np" + str(i) + "_Fraud"

		print "tp rate = ", tp(data, i, threshold) / float(counts.ix[1])
		print "fp rate = ", fp(data, i, threshold) / float(counts.ix[0])
		print "tn rate = ", tn(data, i, threshold) / float(counts.ix[0])
		print "fn rate = ", fn(data, i, threshold) / float(counts.ix[1])