from NeuralNetwork import NeuralNetwork
import json

nn = NeuralNetwork([16, 16, 4], 4)

print "Total training epochs: ", nn.stats['trainingEpochs']

name = 'benzema'

nn.train(
	verbose=False,
	progress=True,
	save=True,
	maxepochs=10000,
	batch=20,
	filename=name,
	autosave=100,
	savestats=True
)

print json.dumps(nn.batchplay(n=100, progress=True), indent=2)
