from NeuralNetwork import NeuralNetwork
import json

nn = NeuralNetwork([16, 16, 16, 4], 4)

print "Total training epochs: ", nn.stats['trainingEpochs']

for i in range(100):
	nn.train(
		verbose=False,
		progress=True,
		save=True,
		maxepochs=100000,
		batch=10,
		filename='bernard',
		autosave=100,
		savestats=True
	)
	print json.dumps(nn.batchplay(n=100, progress=True), indent=2)
