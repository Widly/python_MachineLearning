
#score = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
score = [0,0,1,1, 1,1,1,1, 1,1]
weight = 1.0

decr = 0.02
incr = decr / 4.0

for i in range(0,9):
	if (score[i] > 0.8) and weight < 1.0:
		weight += incr
	else:
		weight -= decr

	if (weight > 1.0):
		weight = 1.0

	print weight