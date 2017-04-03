from NeuralNetwork import NeuralNetwork
import json

nn = NeuralNetwork([16, 16, 16, 4], 4)
# print json.dumps(nn.batchplay(n=100, progress=True), indent=2)

# nn = NeuralNetwork.load()

print "Total training epochs: ", nn.stats['trainingEpochs']

for i in range(100):
	nn.train(verbose=False, progress=True, save=True, maxepochs=10, batch=1000, prefix='bernard_7-')
	print json.dumps(nn.batchplay(n=100, progress=True), indent=2)
