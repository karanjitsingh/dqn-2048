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
	load_mode.add_argument("--model-id", help="Name of saved model")
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


def get_model_id():
	return input("New model id: ")


def cli_options(cli):
	if cli.no_gpu:
		disable_gpu()

	global config
	if cli.json_config:
		config = load_json(cli.json_config)
		print "\'" + cli.json_config + "\':"
		print json.dumps(default_config, indent=2)
	else:
		print "\nUsing default config: "
		print json.dumps(default_config, indent=2)
		config = default_config


def train_new(cli):
	cli_options(cli)
	Train.MiniBatchTrain(config)


def train_saved(cli):

	if cli.model_id:
		if not os.path.isdir("./models/" + cli.model_id):
			print "No such model " + cli.model_id
			exit(1)

		config['model-id'] = cli.model_id

		Train.MiniBatchTrain(config, load_model=True)
	else:
		print "Choose saved model:"

		models = os.listdir('./models')

		for i, id in enumerate(models):
			print "[" + str(i) + "]\t" + id
		index = -1
		while index < 0 or index >= len(models):
			index = input(": ")
		config['model-id'] = models[index]

	cli_options(cli)

	Train.MiniBatchTrain(config, load_model=True)


if cli.cmd == "new":
	import Train
	train_new(cli)
elif cli.cmd == "load":
	import Train
	train_saved(cli)
elif cli.cmd == "default-config":
	print default_config
elif cli.cmd == "json-help":
	print_json_help()