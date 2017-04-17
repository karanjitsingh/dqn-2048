from NeuralNetwork import NeuralNetwork
from functions import ActivationFunctions
from functions import Gradients
import sys

import json
from datetime import datetime
from functions import ActivationFunctions


# Pickle fix
sys.modules['ActivationFunctions'] = ActivationFunctions
sys.modules['Gradients'] = Gradients

name = 'hawking'
# nn = NeuralNetwork.load('./trainlogs/' + name + '.nn')

nn = NeuralNetwork([32, 32, 4], 4, afn=ActivationFunctions.Sigmoid)
#
nn.epsilon = Gradients.Const(0.1)

print "Training \"" + name + "\""
print "Total training epochs: ", nn.stats['trainingEpochs']

print "\n", datetime.now(), "\n"

nn.train(
	verbose=False,
	progress=True,
	save=True,
	maxepochs=200000,
	batch=30,
	replay_size=1000000,
	filename=name,
	autosave=100,
	savestats=True,
)

print json.dumps(nn.batchplay(n=100, progress=True), indent=2)
