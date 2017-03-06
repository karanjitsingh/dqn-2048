from Game import Game
from NeuralNetwork import NeuralNetwork

game = Game(2)
nn = NeuralNetwork([4, 4, 4, 4])
nn.print_network(layers=False)

# game.printgrid()
# print ""
#
# while 1:
#
# 	confidence = nn.propagate(game.grid_to_input()).tolist()
# 	valid, full, halt, score = game.acceptinput(direction=confidence)
# 	game.printgrid()
# 	print "Score: ", score, "\n"
#
# 	if halt or not valid:
# 		break


nn.train(game, verbose=True)
