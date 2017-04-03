from NeuralNetwork import NeuralNetwork
import json
from datetime import datetime

nn = NeuralNetwork([16, 16, 16, 4], 4)
name = 'hawking'

print "Training \"" + name + "\""
print "Total training epochs: ", nn.stats['trainingEpochs']

print "\n", datetime.now(), "\n"

nn.train(
	verbose=False,
	progress=True,
	save=True,
	maxepochs=5000000,
	batch=20,
	replay_size=100000,
	filename=name,
	autosave=5000,
	savestats=True
)

print json.dumps(nn.batchplay(n=100, progress=True), indent=2)
