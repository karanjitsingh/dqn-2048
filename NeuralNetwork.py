import numpy as np
import ActivationFunctions
import Game


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

	def reward(self, input, confidence, output, move_score, valid):
		if valid:
			return move_score
		else:
			return -10

	def train(self, game, verbose=False, log=False):

		halt = False
		i = 0

		if verbose:
			print "Starting simulation..."
			print "i: ", i
			print game.printgrid(), "\n"

		while not halt and i < 10:
			i += 1
			input = game.grid_to_input()

			confidence = self.propagate(input).tolist()
			move_score = game.score

			valid, full, halt, score = game.acceptinput(direction=confidence)

			move_score = score - move_score
			output = game.grid_to_input()

			reward = self.reward(input, confidence, output, move_score, valid)

			''' Discounted Reward '''


			if verbose:
				print "i:", i
				game.printgrid()
				print "Score: ", game.score, "\n"

		if verbose:
			print "Final Network:"
			self.print_network()

# layers = [16,16,16,16,16,4]
# nn = NeuralNetwork(layers)
# nn.print_network(layers=False)
#
# print nn.propagate([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
