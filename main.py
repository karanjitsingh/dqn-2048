import ast
import os
import sys
import argparse
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def disable_gpu():
	os.environ['CUDA_VISIBLE_DEVICES'] = ''


def print_json_help():
	def ws(n, str):
		return str + (' ' * (n-len(str)))

	print ws(20, "key"), "description"
	print "", ws(20, "architecture"), "[\"fcn\"/\"cnn\", ...]"
	print "", ws(20, "learning-rate"), "Learning rate of model"
	print "", ws(20, "discount-factor"), "Discount factor of model"
	print "", ws(20, "epochs"), "Number of games to run"
	print "", ws(20, "loss"), "Loss function"
	print "", ws(20, "replay-size"), "Replay memory size"
	print "", ws(20, "batch-size"), "Mini batch size"
	print "", ws(20, "exploration"), "egreedy/softmax"
	print "", ws(20, "update-mode"), "Update mode for q learning formula"
	print "", ws(20, "epsilon-params"), "Exploration parameter [start, stop, percent steps]"


def parse_cli():
	parser = argparse.ArgumentParser( )

	subparsers = parser.add_subparsers(dest='cmd')

	new_mode = subparsers.add_parser("new", help='Train a new model')
	new_mode.add_argument("--json-config", help="Path to JSON config file")
	new_mode.add_argument("--no-gpu", action='store_true', help="Force training on CPU")

	load_mode = subparsers.add_parser("load", help="Train saved model")
	load_mode.add_argument("--path", "Path to saved model")
	load_mode.add_argument("--json-config", help="Path to JSON config file")
	load_mode.add_argument("--no-gpu", action='store_true', help="Force training on CPU")

	subparsers.add_parser("default-config", help="Print default config")
	subparsers.add_parser("json-help", help="JSON config help")

	return parser.parse_args()


def load_json(path):
	try:
		return json.load(open(path))
	except ValueError:
		print "Problem loading json file."
		exit(1)

cli = parse_cli()

default_config = {
	"architecture": ["cnn"],
	"learning-rate": 0.01,
	"discount-factor": 0.99,
	"epochs": 100000,
	"loss": "huber",
	"replay-size": 100000,
	"batch-size": 10,
	"exploration": "egreedy",
	"update-mode": "all",
	"epsilon-params": [1, 0.1, 15]
}

config = None


if cli.cmd == "new" or cli.cmd == "load":
	if cli.no_gpu:
		disable_gpu()
	if cli.json_config:
		config = load_json(cli.json_config)
		print "\'" + cli.json_config + "\':"
		print json.dumps(default_config, indent=2)
	else:
		print "Using default config: "
		print json.dumps(default_config, indent=2)
		config = default_config
	if cli.cmd == "load":
		print cli

	print ""
elif cli.cmd == "default-config":
	print default_config
elif cli.cmd == "json-help":
	print_json_help()

#
#
# if len(sys.argv) == 1:
# 	print "Using default config: ", default_config
# 	config = default_config
# elif len(sys.argv) == 2:
# 	json_path = sys.argv[1]
# 	data = None
# 	try:
# 		config = json.load(open(json_path))
# 	except ValueError:
# 		print "Couldn't load JSON:", ValueError.message
# 		exit(1)
#
# 	if set(data.keys()) != set(default_config.keys()):
# 		print "Invalid JSON"
# 		exit(1)
#
# 	print "Using config: ", config
# else:
# 	print "Using given arguments: ", sys.argv
# 	config = parse_cli()
#
#
# #
# # def save_model():
# #
# # def load_model():
# #
# #