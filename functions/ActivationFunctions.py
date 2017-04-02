from numpy import exp


# Sigmoid function
class Sigmoid(float or list):
	def __new__(cls, x):
		if type(x) == list:
			return [1 / (1 + exp(-i)) for i in x]
		else:
			return 1 / (1 + exp(-x))

	@staticmethod
	def delta(x):
		return Sigmoid(x) * (1 - Sigmoid(x))
