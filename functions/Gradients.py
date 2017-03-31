import numpy as np


def exponenetial(step):
	return 0.99 * np.exp(np.log(0.05) * step)


def linear(step):
	return 0.99 - 0.94 * step
