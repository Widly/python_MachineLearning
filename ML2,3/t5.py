from t1 import *
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

def vote(ES, thresholds):
	length = float(len(thresholds))
	out = []
	for row in ES:
		if len([pair for pair in zip(row, thresholds) if pair[0] >= pair[1]]) / length >= 0.5:
			out.append(True)
		else:
			out.append(False)

	return out


def Ensemble(data, thresholds):
	to_append = vote(data[["p1_Fraud", "p2_Fraud", "p3_Fraud"]], thresholds)


data = read_csv("vyygrusska.csv", ";")

counts = Series.value_counts(data.CLASS)

print Ensemble(data, [0.5, 0.5, 0.5])