import numpy as np
import ActivationFunctions
from Game import State
from Game import Game



class NeuralNetwork:
	def __init__(self, layers, afn=ActivationFunctions.sigmoid):
		self.layers = layers
		self.network = []
		self.biases = np.random.rand(len(layers))
		self.aFn = afn

		# Weights for input layer (layer 0)
		self.network.append(np.random.rand(layers[0], layers[0]))

		for i in range(len(layers) - 1):
			singlelayer = np.array([])
			for x in range(layers[i]):
				rand = np.random.rand(layers[i + 1])
				singlelayer = np.vstack([singlelayer, rand]) if singlelayer.size else np.array([rand])
			self.network.append(np.transpose(singlelayer))

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
	def propagate(self, input):
		inputmatrix = np.array(input)
		for i in range(len(self.layers)):
			output = np.array([])
			for x in range(self.layers[i]):
				output = np.append(output, self.aFn(self.biases[i] + np.dot(inputmatrix, self.network[i][x])))
			inputmatrix = output

		return output

	@staticmethod
	def reward(fromstate, tostate):
		return tostate.score - fromstate.score

	def train(self, game, verbose=False, log=False):
		halt = False
		i = 0

		if verbose:
			print "Starting simulation..."
			print "i: ", i
			print game.printgrid(), "\n"

		while not halt and i < 1024:
			i += 1

			currstate = game.currState

			input = game.grid_to_input()
			output = self.propagate(input).tolist()
			nextstate = game.transition(direction=output)

			reward = NeuralNetwork.reward(currstate, nextstate)

			score = nextstate.score
			qset = [NeuralNetwork.reward(nextstate, Game.get_next_state(nextstate, Game.getdirection(j))) for j in range(4)]

			output = self.propagate(game.grid_to_input()).tolist()
			qmax = (max(qset), qset.index(max(qset)))
			index = output.index(max(output))
			qsel = (qset[index], index)

			if verbose:
				print "i:", i
				game.printgrid()
				print "Reward: ", reward
				print "Score: ", score
				print "qsel: ", qsel, " qmax: ", qmax
				print ""

		# if verbose:
		# 	print "Final Network:"
		# 	self.print_network()


game = Game(4)
nn = NeuralNetwork([16, 16, 16, 16, 4])
nn.train(game, verbose=True)

