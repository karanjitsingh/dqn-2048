from Game import Game
from NeuralNetwork import NeuralNetwork

game = Game(4)
nn = NeuralNetwork([16, 16, 4])
nn.print_network(layers=False)

while 1:

	confidence = nn.propagate(game.grid_to_input()).tolist()
	valid, full, halt = game.acceptinput(direction=confidence)
	game.printgrid()
	print ""

	if halt or not valid:
		break;
