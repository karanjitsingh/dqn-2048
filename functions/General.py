import numpy as np


def normalize(v):
	mag = np.sqrt(np.sum(np.array(v) ** 2))
	return map(lambda x: x/mag, v)


def reward(fromstate, tostate):
	if not tostate.valid:
		return 0
	elif tostate.empty_tiles >= fromstate.empty_tiles:
		return 1
	return 0
