from NeuralNetwork import NeuralNetwork
import json

# nn = NeuralNetwork([16, 16, 16, 4], 4)
# print json.dumps(nn.batchplay(n=100, progress=True), indent=2)

nn = NeuralNetwork.load()

print "Total training epochs: ", nn.stats['trainingEpochs']

print json.dumps(nn.batchplay(n=100, progress=True), indent=2)

for i in range(100):
	nn.train(verbose=False, progress=True, save=True, maxepochs=100, prefix='bernard_6-')
	print json.dumps(nn.batchplay(n=100, progress=True), indent=2)