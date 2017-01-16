import numpy as np
from pandas import DataFrame

dtype = [('a', 'float32'), ('b', 'float32'),('c', 'float32'),('d', 'float32'),('e', 'float32'), ('penalty', 'float32')]
x = []
x.append([1.0, 2.0, 1.0, 4.0, 5.0, 10.0])
x.append([1.0, 2.0, 9.0, 4.0, 5.0, 20.0])

df = DataFrame(data = x, columns = ['a', 'b', 'c', 'd', 'e', 'penalty'])
df.to_csv('azaza', index = False, sep = ';')