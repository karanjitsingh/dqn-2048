import numpy as np
import ActivationFunctions
import random
from Game import State
from Game import Game
import math

gamma = 0.9												# Discounted reward constant
alpha = 0.5												# Learning rate
e_scale = 0.9											# Lambda or eligibility scale


def epsilon(step):										# Epsilon-greedy selection
	return 0.99 * math.exp(-1 * math.log(0.99/0.05) * step)


class NeuralNetwork:

	def __init__(self, layers, afn=ActivationFunctions.Sigmoid):

		self.layers = layers
		self.network = []
		self.biases = np.random.rand(len(layers))
		self.aFn = afn
		self.depth = len(layers)

		# Weights for input layer (layer 0)
		self.network.append(np.random.rand(layers[0], layers[0]))

		for i in range(self.depth - 1):
			singlelayer = np.array([])
			for x in range(layers[i]):
				rand = np.random.rand(layers[i + 1])
				singlelayer = np.vstack([singlelayer, rand]) if singlelayer.size else np.array([rand])
			self.network.append(np.transpose(singlelayer))

		# Weights are of the form wij from layer j to layer i
		# w11 w12 w13
		# w21 w22 w23
		# w31 w32 w33
		# print self.network

	@staticmethod
	def reward(fromstate, tostate):
		if not tostate.valid:
			return -1
		elif tostate.score - fromstate.score > 0:
			return math.log(tostate.score - fromstate.score, 2)
		return 0

	def print_network(self, biases=True, layers=True):
		print "Layers ", self.layers, "\n"
		if biases:
			print "Biases"
			print self.biases, "\n"
		if layers:
			for i in range(len(self.network)):
				print "Layer ", i
				print self.network[i], "\n"

	# Feedforward propagation
	def propagate(self, input, returnActivations=False):
		inputmatrix = np.array(input)
		output = np.array([])
		multioutput = []
		net = []
		activation = []
		for i in range(len(self.layers)):
			output = np.array([])
			for x in range(self.layers[i]):
				output = np.append(output, self.aFn(self.biases[i] + np.dot(inputmatrix, self.network[i][x])))
			inputmatrix = output

			if returnActivations:
				activation.append(inputmatrix)

		if returnActivations:
			return activation
		else:
			return output

	def tdlearn(self, qsel, reward, e_trace, game, input):			# Temporal difference back propagation
		activation = self.propagate(game.grid_to_input(), True)
		qset = activation[-1].tolist()

		qmax = (max(qset), qset.index(max(qset)))
		td_error = reward + gamma * qmax[0] - qsel[0]

		# Calculate deltas
		delta = []
		delta.insert(0, np.diag(map(lambda x: x*(1-x), activation[-1])))

		for i in range(len(self.layers)-2, -1, -1):
			del_activation_matrix = np.diag(map(lambda x: x*(1-x), activation[i]))
			weights = self.network[i+1];
			product = np.matmul(del_activation_matrix, weights.transpose())
			delta_i = np.matmul(product, delta[0])
			delta.insert(0, delta_i)

		activation.insert(0, np.array(input))

		for i in range(len(delta)):
			activation_3d = np.array([np.diag([activation[i][k]] * delta[i][0].size) for k in range(activation[i].size)])
			e_trace[i] = np.matmul(delta[i],activation_3d)


	def train(self, game, verbose=False, log=False, total=1):
		# Create empty e-trace
		e_trace = []
		output_neurons = self.layers[-1]
		for i in range(len(self.layers)):
			e_trace.append(np.zeros((output_neurons,) + self.network[i].shape))

		halt = False
		i = 0
		if verbose:
			print "Starting simulation..."
			print "i: ", i
			print game.printgrid(), "\n"

		state = game.currState
		qset = self.propagate(game.grid_to_input()).tolist()

		while not halt and i <= total:
			i += 1

			if random.random() < epsilon(i/total):
				index = random.randint(0, 3)		# Choose random action
			else:
				index = qset.index(max(qset))		# Choose policy action

			qsel = (qset[index], index)

			input = game.grid_to_input()
			next_state = game.transition(direction=qset)
			reward = NeuralNetwork.reward(state, next_state)
			state = game.currState

			# TD Learning
			self.tdlearn(qsel, reward, e_trace, game, input)

			if verbose:
				print "i:", i
				game.printgrid()
				print "Reward: ", reward
				print "Score: ", game.currState.score
				print ""

			# if verbose:
			# 	print "Final Network:"
			# 	self.print_network()

game = Game(4)
nn = NeuralNetwork([16, 16, 4])
nn.train(game, verbose=True)

