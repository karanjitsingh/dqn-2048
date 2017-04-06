from datetime import datetime
import json
import pickle
import random
import numpy as np
import sys
import math

from os import listdir
from Game import Game
from components.ProgressBar import ProgressBar
from functions import ActivationFunctions
from functions import Gradients

# Pickle fix
sys.modules['ActivationFunctions'] = ActivationFunctions
sys.modules['Gradients'] = Gradients


def normalize(v):
	return map(lambda x: math.log(x,2) if x else x, v)

class NeuralNetwork:

	def __init__(self, layers, gamedim, afn=ActivationFunctions.Sigmoid):
		# Constants
		self.gamma = 0.8			# Discounted reward constant
		self.alpha = 0.4			# learning rate
		self.epsilon = Gradients.Exponential(1, 0.05)

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
		self.network.append(np.random.rand(layers[0], gamedim*gamedim)*2 - 1)

		for i in range(self.depth - 1):
			singlelayer = np.array([])
			for x in range(layers[i]):
				rand = np.random.rand(layers[i + 1]) * 2 - 1
				singlelayer = np.vstack([singlelayer, rand]) if singlelayer.size else np.array([rand])
			self.network.append(np.transpose(singlelayer))

		# Create empty e-trace
		self.e_trace = []
		output_neurons = self.layers[-1]

		# e_ijk = e[k][j][i]
		for i in range(self.depth):
			self.e_trace.append(np.ndarray(self.network[i].shape[::-1] + (output_neurons,)))

	def save(self, path='trainlogs/', filename='default'):
		path += filename + ".nn"

		with open(path, "wb") as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	def savestats(self, path='trainlogs/', filename='default'):
		path += filename + ".stats"
		stat = self.batchplay(n=100, progress=False)
		stat = "Training Epochs:" + str(self.stats['trainingEpochs']) + "\n" + json.dumps(stat, indent=4) + "\n\n"
		print "\n" + stat
		with open(path, "a+") as output:
			output.write(stat)

	@staticmethod
	def savereplay(memory, path='trainlogs/', filename='default'):
		path += filename + ".replay"
		with open(path, "wb") as output:
			pickle.dump(memory, output, pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def loadreplay(path='trainlogs/', filename='default'):
		path += filename + ".replay"
		with open(path, "rb") as input:
			return pickle.load(input)

	@staticmethod
	def load(path=''):
		if path == '':
			files = filter(lambda x: x[-3:] == ".nn", listdir('./trainlogs'))
			files.sort()
			f = "./trainlogs/" + files[-1]
		else:
			f = path

		print "[", f, "]"

		with open(f, "rb") as input:
			return pickle.load(input)

	@staticmethod
	def reward(fromstate, tostate):
		if not tostate.valid:
			return -1
		else:
			x = 0
			y = 0
			for i in range(len(fromstate.grid)):
				for j in range(len(fromstate.grid)):
					if not fromstate.grid[i][j]:
						x += 1
					if not tostate.grid[i][j]:
						y += 1
			if y >= x:
				return 1

			# return 1
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
		input = np.array(input)
		inputmatrix = np.array(self.aFn(input))
		output = np.array([])
		activation = []
		for i in range(len(self.layers)):
			output = np.array([])
			for x in range(self.layers[i]):
				output = np.append(output, self.aFn(np.dot(inputmatrix, self.network[i][x]) + self.biases[i][x]))
			inputmatrix = output

			if returnActivations:
				activation.append(inputmatrix)

		if returnActivations:
			return activation
		else:
			return output

	def td_gradient(self, qset, reward, next_input, sel_index, input):			# Temporal difference back propagation
		activation = self.propagate(normalize(next_input), True)
		qmax = max(activation[-1].tolist())

		# td_error = reward + self.gamma * np.array(qmax) - qset
		td_error = map((lambda i: (reward + self.gamma * np.array(qmax) - qset[sel_index])*2 if i == sel_index else 0), qset)

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
			self.e_trace[i] = self.alpha * np.array([delta[i] * activation[i][k] for k in range(activation[i].size)])

		delta_biases = []
		delta_weights = []

		# Update biases
		for i in range(self.depth):
			delta_biases.append(self.alpha * np.matmul(delta[i], td_error))

		# Update weights
		for i in range(self.depth):
			del_weights = np.zeros(self.network[i].shape)
			for k in range(self.e_trace[i].shape[0]):
				for j in range(self.e_trace[i].shape[1]):
					# Error for each output neuron
					del_weights[j][k] = np.matmul(self.e_trace[i][k][j], td_error)

					# Error for single output neuron
					# del_weights[j][k] = del_w[i][k][j][qsel[1]] * td_error[qsel[1]]

			delta_weights.append(del_weights)

		return delta_weights, delta_biases

	def learn(self, del_w, del_b):
		# Update biases
		for i in range(self.depth):
			self.biases[i] += del_b[i]

		# Update weights
		for i in range(self.depth):
			self.network[i] += del_w[i]

	def train(self, maxepochs=1000, batch=10, replay_size=1000000, verbose=False, progress=False, save=False, filename='', autosave=100, savestats=False):

		batch_b = [np.array(np.zeros(i)) for i in self.layers]
		batch_w = list()
		batch_w.append(np.random.rand(self.layers[0], self.gamedim * self.gamedim) * 2 - 1)
		for i in range(self.depth - 1):
			singlelayer = np.array([])
			for x in range(self.layers[i]):
				rand = np.zeros(self.layers[i + 1])
				singlelayer = np.vstack([singlelayer, rand]) if singlelayer.size else np.array([rand])
			batch_w.append(np.transpose(singlelayer))

		replay = []
		epochs = 0
		if self.stats['trainingEpochs'] > 0:
			epochs = self.stats['trainingEpochs']
			replay = self.loadreplay(filename=filename)

		print len(replay)
		pbar = None


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
				qset = self.propagate(normalize(game.grid_to_input())).tolist()

				if random.random() < self.epsilon(float(epochs)/maxepochs):
					index = random.randint(0, 3)		# Choose random action
				else:
					index = qset.index(max(qset))		# Choose policy action

				input = normalize(game.grid_to_input())
				next_state = game.transition(direction=index)
				reward = NeuralNetwork.reward(state, next_state)
				state = game.currState
				halt = state.halt


				# Add transition to experience
				replay.append((qset, reward, game.grid_to_input(), index, input, ))
				if len(replay) > replay_size:
					replay.pop(0)

				if verbose:
					print "i:", i
					game.printgrid()
					print "Reward: ", reward
					print "Score: ", game.currState.score
					print ""

			# Learn from epoch / experience replay
			if epochs > 1:
				for j in range(batch):
					index = random.randint(0, len(replay) - 1)

					# Generate gradients with TD backpropagation
					dw, db = self.td_gradient(*(replay[index]))

					# Accumulate gradients and form mini-batch
					for i in range(self.depth):
						batch_b[i] += db[i]
						batch_w[i] += dw[i]

			# Learn from minibatch
			self.learn(batch_w, batch_b)
			for i in range(self.depth):
				batch_b[i].fill(0)
				batch_w[i].fill(0)

			epochs += 1

			# Update progress bar
			if progress:
				if pbar is None:
					pbar = ProgressBar(40, maxepochs, "Epochs", verbose)
				pbar.update(epochs)

			if not epochs%autosave:
				self.stats['trainingEpochs'] += autosave
				self.save(filename=filename)
				self.savereplay(replay, filename=filename)
				if savestats:
					self.savestats(filename=filename)

			if i % batch:
				self.learn(batch_w, batch_b)

		if save:
			self.stats['trainingEpochs'] += maxepochs % autosave
			self.save(filename=filename)
			self.savereplay(replay, filename=filename)
			if savestats:
				self.savestats(filename=filename)

	def play(self, verbose=False):
		stat = {}
		game = Game(self.gamedim)
		i = 0
		dtext = ['down', 'right', 'up', 'left']
		invalid = {'count': 0, 'offset': 0}

		if verbose:
			print "New game..."
			print "i: ", i
			print game.printgrid()

		state = game.currState

		while not state.halt:
			i += 1

			qset = self.propagate(normalize(game.grid_to_input())).tolist()

			policy = [k for k in enumerate(qset)]
			policy.sort(key=lambda x: x[1])
			policy = map(lambda x: x[0], policy[::-1])

			state = game.transition(direction=policy[invalid['offset']])

			d = dtext[policy[invalid['offset']]]

			if not state.valid:
				invalid['count'] += 1
				invalid['offset'] += 1
			else:
				invalid['offset'] = 0

			if verbose:
				print "i:", i
				print d
				game.printgrid()
				print "Score: ", game.currState.score
				print "Valid: ", state.valid, "\t Halt: ", state.halt
				print ""

		stat['maxTile'] = max([max(game.currState.grid[k]) for k in range(len(game.currState.grid))])
		stat['minTile'] = min([min(filter(lambda x: x, game.currState.grid[k])) for k in range(len(game.currState.grid))])
		stat['score'] = game.currState.score
		stat['steps'] = i
		stat['invalid'] = invalid['count']
		stat['grid'] = game.currState.grid

		if verbose:
			print json.dumps(stat, indent=2)

		return stat

	def batchplay(self, n=1, progress=False, verbose=False):
		avgstat = {
			'totalGames': n,
			'maxTileCount': {},
			'avgScore': 0,
			'avgSteps': 0,
			'avgInvalid': 0,
			'minScore': 0,
			'maxScore': 0,
			'datetime': str(datetime.now())
		}

		games = []

		for i in range(n):
			stat = self.play(verbose=verbose)

			if verbose:
				print json.dumps(stat, indent=2)

			games.append(stat)

			if str(stat['maxTile']) in avgstat['maxTileCount'].keys():
				avgstat['maxTileCount'][str(stat['maxTile'])] += 1
			else:
				avgstat['maxTileCount'].update({str(stat['maxTile']): 1})

			if stat['score'] < avgstat['minScore'] or not avgstat['minScore']:
				avgstat['minScore'] = stat['score']

			if stat['score'] > avgstat['maxScore']:
				avgstat['maxScore'] = stat['score']

			avgstat['avgScore'] += stat['score']/float(n)
			avgstat['avgSteps'] += stat['steps']/float(n)
			avgstat['avgInvalid'] += stat['invalid']/float(n)

			if progress:
				if i == 0:
					p = ProgressBar(40, n, "Games", verbose)
				p.update(i+1)
		return avgstat


