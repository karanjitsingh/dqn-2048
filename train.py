from NeuralNetwork import NeuralNetwork
import json
from datetime import datetime
from functions import ActivationFunctions

name = 'hope'
nn = NeuralNetwork.load('./trainlogs/' + name + '.nn')
#
# nn = NeuralNetwork([16, 16, 16, 16, 4], 4, afn=ActivationFunctions.Sigmoid)

print "Training \"" + name + "\""
print "Total training epochs: ", nn.stats['trainingEpochs']

print "\n", datetime.now(), "\n"

nn.train(
	verbose=False,
	progress=True,
	save=True,
	maxepochs=1000000,
	batch=30,
	replay_size=100000,
	filename=name,
	autosave=2500,
	savestats=True,
)

print json.dumps(nn.batchplay(n=100, progress=True), indent=2)
