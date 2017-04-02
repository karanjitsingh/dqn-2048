from NeuralNetwork import NeuralNetwork
import time
import json

nn = NeuralNetwork([16, 16, 4], 4)
total = 0

for i in range(500):
	print "i: ", i+1
	start = time.time()

	print "Training epochs", nn.stats['trainingEpochs']
	nn.train(maxepochs=100, progress=True, save=True)
	print json.dumps(nn.batchplay(n=100, progress=True), indent=2)
	time_taken = time.time() - start
	total += time_taken
	print time_taken, " Seconds"

	eta = (1000-(i+1)) * total / (i+1)
	s = eta % 60
	m = 0
	h = 0

	if (eta-s)/60 > 1:
		m = (eta-s) / 60

	if (m-(m%60))/60 > 1:
		h = int(m/60)
		m %= 60

	print "ETA ", str(int(h)) + ":" + str(int(m)) + ":" + str(int(s)), "\n"
