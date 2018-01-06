import numpy as np


def normalize(v):
	# mag = np.sqrt(np.sum(np.array(v) ** 2))
	return map(lambda x: 0 if x == 0 else np.log2(x)/10.0, v)


def reward(fromstate, tostate):
	if not tostate.valid:
		return 0
	elif tostate.score > fromstate.score:
		return 1
	return 0
