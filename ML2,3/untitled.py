from pandas import read_csv, Series
import numpy as np

data = read_csv("vyygrusska.csv", ";")
data = data[np.logical_and(data["CLASS"] != "U", True)].reset_index(drop = True)

Time = data["EVENT_TIME"]
print Time