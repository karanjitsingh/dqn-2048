from NeuralNetwork import NeuralNetwork
import json
from datetime import datetime

name = 'hawking'
# nn = NeuralNetwork.load('./trainlogs/hope2.nn')

nn = NeuralNetwork([256, 4], 4)

print "Training \"" + name + "\""
print "Total training epochs: ", nn.stats['trainingEpochs']

print "\n", datetime.now(), "\n"

nn.train(
	verbose=False,
	progress=True,
	save=True,
	maxepochs=50000,
	batch=20,
	replay_size=100000,
	filename=name,
	autosave=500,
	savestats=True
)

print json.dumps(nn.batchplay(n=100, progress=True), indent=2)
