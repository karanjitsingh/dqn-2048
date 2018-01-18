import ast
import os
import sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

default_args = [
	sys.argv[0],
	"[\"cnn\"]",
	"0.1",
	"0.5",
	"30000",
	"mse",
	"1000",
	"10",
	"egreedy",
	"[1,0.1,15]"
]

print "Using default arguments: ", default_args
sys.argv = default_args


def parse_cli():
	parser = argparse.ArgumentParser()
	parser.add_argument("architecture", help="[\"fcn\"/\"conv\", ...]")
	parser.add_argument("learning-rate", help="Learning rate of model")
	parser.add_argument("discount-factor", help="Discount factor of model")
	parser.add_argument("epochs", help="Number of games to run")
	parser.add_argument("loss", help="Loss function")
	parser.add_argument("replay-size", help="Replay memory size")
	parser.add_argument("batch-size", help="Mini batch size")
	parser.add_argument("exploration", help="egreedy/softmax")
	parser.add_argument("epsilon-params", help="Exploration parameter [start, stop, %steps]")

	string_args = ["exploration", "loss"]

	args = vars(parser.parse_args())

	for key, value in args.iteritems():
		if key not in string_args:
			args[key] = ast.literal_eval(value)

	args["trainer-id"] = raw_input("Trainer id: ")

	return args

#
# def save_model():
#
# def load_model():
#
#