from components.ProgressBar import ProgressBar

from Game import State
from Game import Game
from os import listdir

import numpy as np
import ActivationFunctions
import random
import datetime
import pickle
import json


def normalize(v):
	v = np.array(v)
	return (v/np.sqrt(np.sum(v*v))).tolist()


class NeuralNetwork:
	def __init__(self, layers, gamedim, afn=ActivationFunctions.Sigmoid):
		# Constants
		self.gamma = 0.8			# Discounted reward constant
		self.alpha = 0.7			# Lambda / eligibility scale
		self.epsilon = 0.674		# Epsilon greedy selection
		# self.epsilon = lambda step: 0.99 * np.exp(np.log(0.05) * step)			# Decreasing exponential epsilon

		# Game settings
		self.gamedim = gamedim

		# NN Architecture
		self.layers = layers
		self.network = []
		self.biases = [np.array(np.random.rand(i) * 2 - 1) for i in layers]

		self.aFn = afn
		self.depth = len(layers)

		self.stats = dict()
		self.stats['trainingEpochs'] = 0

		# Weights for input layer (layer 0)
		self.network.append(np.random.rand(layers[0], layers[0])*2 - 1)

		for i in range(self.depth - 1):
			singlelayer = np.array([])
			for x in range(layers[i]):
				rand = np.random.rand(layers[i + 1]) * 2 - 1

				singlelayer = np.vstack([singlelayer, rand]) if singlelayer.size else np.array([rand])
			self.network.append(np.transpose(singlelayer))

	def save(self):
		now = datetime.datetime.now()
		path = "trainlogs/" + now.strftime('%y-%m-%d-%I-%M.nn')

		with open(path, "wb") as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load(path=''):
		if path == '':
			files = filter(lambda x: x[-3:] == ".nn", listdir('./trainlogs'))
			files.sort()
			f = "./trainlogs/" + files[-1]
		else:
			f = path

		with open(f, "rb") as input:
			nn = pickle.load(input)
			print "NN Stats\n", nn.stats

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
		input = np.array(normalize(input))
		inputmatrix = np.array(self.aFn(input))
		output = np.array([])
		activation = []
		for i in range(len(self.layers)):
			output = np.array([])
			for x in range(self.layers[i]):
				# if i == len(self.layers)-1:
				# 	print np.dot(inputmatrix, self.network[i][x]), self.aFn(np.dot(inputmatrix, self.network[i][x]))
				output = np.append(output, self.aFn(np.dot(inputmatrix, self.network[i][x])))
			inputmatrix = output

			if returnActivations:
				activation.append(inputmatrix)

		if returnActivations:
			return activation
		else:
			return output

	def tdlearn(self, qset, reward, del_w, game, input):			# Temporal difference back propagation
		activation = self.propagate(game.grid_to_input(), True)
		qmax = max(activation[-1].tolist())

		td_error = reward + self.gamma * np.array(qmax) - qset

		# Calculate deltas
		delta = []
		delta.insert(0, np.diag(map(lambda x: x*(1-x), activation[-1])))

		for i in range(self.depth-2, -1, -1):
			del_activation_matrix = np.diag(map(lambda x: x*(1-x), activation[i]))
			weights = self.network[i+1]
			product = np.matmul(del_activation_matrix, weights.transpose())
			delta_i = np.matmul(product, delta[0])
			delta.insert(0, delta_i)

		input = self.aFn(input)
		activation.insert(0, np.array(input))

		# Calculate eligibility traces
		for i in range(len(delta)-1, -1, -1):
			# Online training
			del_w[i] = self.alpha * np.array([delta[i] * activation[i][k] for k in range(activation[i].size)])

		# Update weights
		for i in range(self.depth):
			del_weights = np.zeros(self.network[i].shape)
			for k in range(del_w[i].shape[0]):
				for j in range(del_w[i].shape[1]):
					# Error for each output neuron
					del_weights[j][k] = np.matmul(del_w[i][k][j], td_error)

					# Error for single output neuron
					# del_weights[j][k] = del_w[i][k][j][qsel[1]] * td_error[qsel[1]]

			self.network[i] += del_weights

	def train(self, maxepochs=1000, verbose=False, progress=False, save=False):
		# Create empty e-trace
		del_w = []
		output_neurons = self.layers[-1]

		# e_ijk = e[k][j][i]
		for i in range(self.depth):
			del_w.append(np.zeros(self.network[i].shape[::-1] + (output_neurons,)))

		epochs = 0
		while epochs < maxepochs:
			game = Game(self.gamedim)
			halt = False
			if verbose:
				print "New game..."
				print "i: ", i
				print game.printgrid(), "\n"

			state = game.currState

			i = 0
			while not halt:
				i += 1
				qset = self.propagate(game.grid_to_input()).tolist()

				if random.random() < self.epsilon:
					index = random.randint(0, 3)		# Choose random action
				else:
					index = qset.index(max(qset))		# Choose policy action

				# print qset

				input = normalize(game.grid_to_input())

				next_state = game.transition(direction=index)
				reward = NeuralNetwork.reward(state, next_state)
				state = game.currState
				halt = state.halt

				# TD Learning
				self.tdlearn(qset, reward, del_w, game, input)

				if verbose:
					print "i:", i
					game.printgrid()
					print "Reward: ", reward
					print "Score: ", game.currState.score
					print ""
			epochs += 1

			if progress:
				if epochs == 1:
					p = ProgressBar(40, maxepochs, "Epochs", verbose)
				p.update(epochs)

		if save:
			self.stats['trainingEpochs'] += maxepochs
			self.save()

	def play(self, verbose=False):
		stat = {}
		game = Game(self.gamedim)
		i = 0
		invalid = {'count': 0, 'offset': 0}

		if verbose:
			print "New game..."
			print "i: ", i
			print game.printgrid()

		state = game.currState

		while not state.halt:
			i += 1

			qset = self.propagate(game.grid_to_input()).tolist()

			policy = np.array(qset)
			policy.sort()
			policy = map(lambda i: qset.index(i), policy[::-1])

			state = game.transition(direction=policy[invalid['offset']])

			if not state.valid:
				invalid['count'] += 1
				invalid['offset'] += 1
			else:
				invalid['offset'] = 0

			if verbose:
				print "i:", i
				game.printgrid()
				print "Score: ", game.currState.score
				print "Valid: ", state.valid, "\t Halt: ", state.halt
				print ""

		stat['maxTile'] = max([max(game.currState.grid[k]) for k in range(len(game.currState.grid))])
		stat['score'] = game.currState.score
		stat['steps'] = i
		stat['invalid'] = invalid['count']
		return stat

	def batchplay(self, n=1, progress=False, verbose=False):
		avgstat = {
			'totalGames': n,
			'maxTileCount': {},
			'avgScore': 0,
			'avgSteps': 0,
			'avgInvalid': 0
		}

		for i in range(n):
			stat = self.play(verbose=verbose)

			if str(stat['maxTile']) in avgstat['maxTileCount'].keys():
				avgstat['maxTileCount'][str(stat['maxTile'])] += 1
			else:
				avgstat['maxTileCount'].update({str(stat['maxTile']): 1})

			avgstat['avgScore'] += stat['score']/float(n)
			avgstat['avgSteps'] += stat['steps']/float(n)
			avgstat['avgInvalid'] += stat['invalid']/float(n)

			if progress:
				if i == 0:
					p = ProgressBar(40, n, "Games", verbose)
				p.update(i+1)
		return avgstat

nn = NeuralNetwork([16, 4], 4)
print json.dumps(nn.batchplay(n=1000, progress=True), indent=2)
print nn.network
nn.train(verbose=False, progress=True, save=True, maxepochs=3000)
print nn.network
print json.dumps(nn.batchplay(n=1000, progress=True), indent=2)
