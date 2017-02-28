import numpy as np
import ActivationFunctions


class NeuralNetwork:

	def __init__(self, layers, afn=ActivationFunctions.sigmoid):
		self.layers = layers
		self.network = []
		self.biases = np.random.rand(len(layers))
		self.aFn = afn

		# Weights for inputs or layer 0
		self.network.append(np.random.rand(layers[0], layers[0]))

		for i in range(len(layers)-1):
			singlelayer = np.array([])
			for x in range(layers[i]):
				rand = np.random.rand(layers[i + 1])
				singlelayer = np.vstack([singlelayer, rand]) if singlelayer.size else np.array([rand])
			self.network.append(np.transpose(singlelayer))

	def print_network(self,biases=True,layers=True):
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
		input = np.array(input)
		for i in range(len(self.layers)):
			output = np.array([])
			for x in range(self.layers[i]):
				output = np.append(output, self.aFn(self.biases[i] + np.dot(input, self.network[i][x])))
			input = output
		return output



# layers = [16,16,16,16,16,4]
# nn = NeuralNetwork(layers)
# nn.print_network(layers=False)
#
# print nn.propagate([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

