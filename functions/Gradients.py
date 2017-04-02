import numpy as np


def exponenetial(step):
	return 0.99 * np.exp(np.log(0.05) * step)


def linear(step):
	return 0.99 - 0.94 * step


class Const:
	def __init__(self, val):
		self.val = val

	def __call__(self, unused):
		return self.val
