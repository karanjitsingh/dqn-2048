from numpy import exp
from numpy import log


# Sigmoid function
class Sigmoid(float or list):
	def __new__(cls, x):
		if type(x) == list:
			return [1/(1+exp(-1*i)) for i in x]
		else:
			return 1/(1+exp(-1*i))

	@staticmethod
	def delta(x):
		return x*(1-x)


class Softplus(float or list):
	def __new__(cls, x):
		if type(x) == list:
			return [log(1 + exp(i)) for i in x]
		else:
			return log(1+exp(x))

	@staticmethod
	def delta(x):
		return 1/(1+exp(-1*x))


class Linear(float or list):
	def __new__(cls, x):
		return x

	@staticmethod
	def delta(x):
		return 1

