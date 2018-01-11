import numpy as np
import random


def normalize(v):
	# mag = np.sqrt(np.sum(np.array(v) ** 2))
	return map(lambda x: 0 if x == 0 else np.log2(x)/10.0, v)


def reward(fromstate, tostate):
	if not tostate.valid:
		return 0
	elif tostate.score > fromstate.score:
		return 1
	return 0


class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, memory_tuple):
		self.memory.insert(0, memory_tuple)

		if len(self.memory) > self.capacity:
			self.memory.pop()

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)
