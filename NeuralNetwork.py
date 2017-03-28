import numpy as np
import ActivationFunctions
import random
from Game import State
from Game import Game
import datetime
import pickle
from os import listdir


def epsilon(step):										# Epsilon-greedy selection
	return 0.99 * np.exp(-1 * np.log(0.99/0.05) * step)


class NeuralNetwork:
	def __init__(self, layers, afn=ActivationFunctions.Sigmoid):

		# Constants
		self.gamma = 0.9  # Discounted reward constant
		self.e_scale = 0.9  # Lambda / eligibility scale

		# NN Architecture
		self.layers = layers
		self.network = []
		self.biases = np.random.rand(len(layers)) * 2 - 1
		self.aFn = afn
		self.depth = len(layers)

		self.stats = {}
		self.stats['trainingEpochs'] = 0

		# Weights for input layer (layer 0)
		self.network.append(np.random.rand(layers[0], layers[0])*2 - 1)

		for i in range(self.depth - 1):
			singlelayer = np.array([])
			for x in range(layers[i]):
				rand = np.random.rand(layers[i + 1]) * 2 -1

				singlelayer = np.vstack([singlelayer, rand]) if singlelayer.size else np.array([rand])
			self.network.append(np.transpose(singlelayer))

	def save(self):
		now = datetime.datetime.now()
		path = "trainlogs/" + now.strftime('%y-%m-%d-%I-%M.nn')

		with open(path, "wb") as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load(path=''):
		file = ''
		if path == '':
			files = filter(lambda x: x[-3:] == ".nn", listdir('./trainlogs'))
			files.sort()
			file = "./trainlogs/" + files[-1]
		else:
			file = path

		with open(file, "rb") as input:
			return pickle.load(input)

	@staticmethod
	def reward(fromstate, tostate):
		if not tostate.valid:
			return -1
		elif tostate.score - fromstate.score > 0:
			return 1
			# return math.log(tostate.score - fromstate.score, 2)
		return 0

	def print_network(self, biases=True, layers=True):
		print "Layers ", self.layers, "\n"
		if biases:
			print "Biases"
			print self.biases, "\n"
		if layers:
			for i in range(self.depth):
				print "Layer ", i
				print self.network[i], "\n"

	# Feedforward propagation
	def propagate(self, input, returnActivations=False):
		inputmatrix = np.array(self.aFn(input))
		output = np.array([])
		multioutput = []
		net = []
		activation = []
		for i in range(len(self.layers)):
			output = np.array([])
			for x in range(self.layers[i]):
				output = np.append(output, self.aFn(np.dot(inputmatrix, self.network[i][x])))
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

		td_error = reward + self.gamma * np.array(qset) - qsel[0]

		# Calculate deltas
		delta = []
		delta.insert(0, np.diag(map(lambda x: x*(1-x), activation[-1])))

		for i in range(self.depth-2, -1, -1):
			del_activation_matrix = np.diag(map(lambda x: x*(1-x), activation[i]))
			weights = self.network[i+1];
			product = np.matmul(del_activation_matrix, weights.transpose())
			delta_i = np.matmul(product, delta[0])
			delta.insert(0, delta_i)

		input = self.aFn(input)
		activation.insert(0, np.array(input))

		# Calculate eligibility traces
		for i in range(len(delta)-1, -1, -1):
			e_trace[i] += self.e_scale * np.array([delta[i] * activation[i][k] for k in range(activation[i].size)])

		# Update weights
		for i in range(self.depth):
			del_weights = np.zeros(self.network[i].shape)
			for k in range(e_trace[i].shape[0]):
				for j in range(e_trace[i].shape[1]):
					del_weights[j][k] = np.matmul(e_trace[i][k][j], td_error)
			self.network[i] += del_weights

	def train(self, gameDim, maxEpochs=1000, verbose=False, progress=False, save=False):
		# Create empty e-trace
		e_trace = []
		output_neurons = self.layers[-1]

		# e_ijk = e[k][j][i]
		for i in range(self.depth):
			e_trace.append(np.zeros(self.network[i].shape[::-1] + (output_neurons,)))

		epochs = 0
		while epochs < maxEpochs:
			game = Game(gameDim)
			halt = False
			i = 0
			if verbose:
				print "New game..."
				print "i: ", i
				print game.printgrid(), "\n"

			state = game.currState
			qset = self.propagate(game.grid_to_input()).tolist()

			while not halt:
				i += 1

				if random.random() < epsilon(epochs/maxEpochs):
					index = random.randint(0, 3)		# Choose random action
				else:
					index = qset.index(max(qset))		# Choose policy action

				qsel = (qset[index], index)

				input = game.grid_to_input()

				next_state = game.transition(direction=qsel[1])
				reward = NeuralNetwork.reward(state, next_state)
				state = game.currState
				halt = state.halt

				# TD Learning
				self.tdlearn(qsel, reward, e_trace, game, input)

				if verbose:
					print "i:", i
					game.printgrid()
					print "Reward: ", reward
					print "Score: ", game.currState.score
					print ""
			epochs += 1

			if progress:
				print "Epochs ", epochs, "/", maxEpochs

		if save:
			self.stats['trainingEpochs'] = maxEpochs
			self.save()

# nn = NeuralNetwork([16, 16, 4])
# nn.train(gameDim=4, verbose=True, maxEpochs=1, progress=True, save=True)
nn = NeuralNetwork.load()
print nn.__dict__