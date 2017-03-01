from Game import Game
from NeuralNetwork import NeuralNetwork

game = Game(4)
nn = NeuralNetwork([16, 16, 4])
nn.print_network(layers=False)

game.printgrid()
print "l"

while 1:

	confidence = nn.propagate(game.grid_to_input()).tolist()
	valid, full, halt, score = game.acceptinput(direction=confidence)
	game.printgrid()
	print "Score: ", score, "\n"

	if halt or not valid:
		break;
