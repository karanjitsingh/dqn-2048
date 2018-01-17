import ast
import os
import sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

default_args = [sys.argv[0], "[16,256,256,256,4]", "0.1", "0.92", "30000", "\"AAA\""]
print "Using default arguments: ", default_args
sys.argv = default_args


def parse_cli():
	parser = argparse.ArgumentParser()
	parser.add_argument("structure", help="Network structure python array")
	parser.add_argument("learning-rate", help="Learning rate of model")
	parser.add_argument("discount-factor", help="Discount factor of model")
	parser.add_argument("epochs", help="Number of games to run")
	parser.add_argument("training-id", help="Number of games to run")

	args = vars(parser.parse_args())
	for key, value in args.iteritems():
		args[key] = ast.literal_eval(value)

	hidden_layers = args["structure"][1:-1]
	trainer_id = args["training-id"]

	return args, hidden_layers, trainer_id