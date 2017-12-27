import math
import numpy as np


def normalize(v):
	# mag = np.sqrt(np.sum(np.array(v) ** 2))
	maxtile = float(max(v))
	return map(lambda x: x/maxtile, v)
	# return map(lambda x: x/mag, v)


def reward(fromstate, tostate):
	if not tostate.valid:
		return -1
	elif tostate.empty_tiles >= fromstate.empty_tiles:

		# reward based on score
		# return np.log2(tostate.score - fromstate.score)/11

		# reward based on number of tiles minimized
		# return float(tostate.empty_tiles)/15

		# reward based on score
		return float(tostate.score - fromstate.score)/512

		# simple reward
		# return 1
	return 0
