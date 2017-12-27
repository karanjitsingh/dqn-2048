import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import shutil
import numpy as np
import random
import tensorflow as tf
from Game import Game
from functions import Gradients
from functions.General import *
import math
import argparse
import ast


# Process command line args
def parse_cli(argv):
	if argv:
		print argv
		sys.argv = argv

	parser = argparse.ArgumentParser()
	parser.add_argument("structure", help="Network structure python array")
	parser.add_argument("learning-rate", help="Learning rate of model")
	parser.add_argument("discount-factor", help="Discount factor of model")
	parser.add_argument("epochs", help="Number of games to run")

	args = vars(parser.parse_args())
	for key, value in args.iteritems():
		args[key] = ast.literal_eval(value)

	hidden_layers = args["structure"][1:-1]

	return args, hidden_layers


# Generate fully connected neural network
def network_architecture(hidden_layers, input_size, output_size):
	inputs = tf.placeholder(shape=[1, input_size], dtype=tf.float32)

	last_layer = input_size
	weights = []
	biases = []
	activation = [inputs]

	for i, n in enumerate(hidden_layers):
		weights.append(tf.Variable(tf.random_normal([last_layer, n], -0.5, 0.5)))
		biases.append(tf.Variable(tf.constant(0.1, shape=[n])))
		activation.append(tf.nn.relu(tf.add(tf.matmul(activation[i], weights[i]), biases[i])))
		last_layer = n

	weights.append(tf.Variable(tf.random_normal([last_layer, output_size], -0.5, 0.5)))
	biases.append(tf.Variable(tf.constant(0.1, shape=[output_size])))

	Qout = tf.nn.relu(tf.add(tf.matmul(activation[-1], weights[-1]), biases[-1]))

	return {
		'input': inputs,
		'weights': weights,
		'biases': biases,
		'output': Qout
	}


params, hidden_layers = parse_cli([sys.argv[0], "[16,256,4]", "0.01", "0.8", "5"])


tf.reset_default_graph()

tf_network = network_architecture(hidden_layers, 16, 4)
Qout = tf_network['output']
inputs = tf_network['input']

Qmean = tf.reduce_mean(Qout)
predict = tf.argmax(Qout, 1)
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=params["learning-rate"])

updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

#  Set learning parameters
y = params["discount-factor"]
e = 0.05

epsilon = Gradients.Exponential(start=0.9, stop=0.01)
num_episodes = int(params["epochs"])

agent_id = "AAB"


def initialize_log(tf_variables, agent_id, path):
	if os.path.exists(path) and os.path.isdir(path):
		shutil.rmtree(path)

	with tf.name_scope(agent_id):
		for key, var in tf_variables.items():
			tf.summary.scalar(key, var)

	return tf.summary.merge_all(), tf.summary.FileWriter(path, graph=tf.get_default_graph())


def add_scalar_summary(values, summary_writer, agent_id, step):
	for key, val in values.items():
		summary = tf.Summary(value=[
			tf.Summary.Value(tag=str(agent_id + "/" + key), simple_value=val),
		])
		summary_writer.add_summary(summary, step)

with tf.Session() as sess:
	sess.run(init)

	summary_op, summary_writer = initialize_log({
		'loss': loss,
		'Qmean': Qmean
	}, agent_id, "./log")

	score = 0
	total_steps = 0
	tile_threshold = 32

	maxtile_history = []
	i = 0
	game_i = 0

	while i < num_episodes:
		# Reset environment and get first new observation
		game = Game(4)
		s = normalize(game.grid_to_input())
		reward_sum = 0
		halt = False
		steps = 0
		rand_steps = 0
		invalid_steps = 0
		# The Q-Network
		while not halt:
			steps += 1
			currstate = game.currState
			# if i == 0:
			# 	game.printgrid()

			# Choose an action by greedily (with e chance of random action) from the Q-network
			a, allQ = sess.run([predict, Qout], feed_dict={inputs: [s]})

			random_action = False
			policy_action = 0
			sorted_action = np.argsort(-np.array(allQ))[0]

			if np.random.rand(1) < e:
				a[0] = random.randint(0, 3)
				rand_steps += 1
				random_action = True
			# 	if i == 0:
			# 		print "random action: ", a[0]
			# 		print ""
			# elif i == 0:
			# 	print "policy action: ", a[0]
			# 	print ""

			nextstate = game.transition(a[0])

			if not nextstate.halt and not nextstate.valid:
				if random_action:
					b = a[0]
					while b == a[0]:
						b = random.randint(0, 3)
					a[0] = b
				else:
					invalid_steps += 1
					policy_action += 1
					a[0] = sorted_action[policy_action]
				nextstate = game.transition(a[0])

			maxtile = max([max(game.currState.grid[k]) for k in range(len(game.currState.grid))])

			# Get new state and reward from environment
			r = reward(currstate, nextstate)
			s1 = normalize(game.grid_to_input())
			halt = nextstate.halt

			if maxtile == tile_threshold * (2 ** i):
				halt = True

			# Obtain the Q' values by feeding the new state through our network
			Q1 = sess.run(Qout, feed_dict={inputs: [s1]})
			# Obtain maxQ' and set our target value for chosen action.
			maxQ1 = np.max(Q1)
			targetQ = allQ
			targetQ[0, a[0]] = r + (y*maxQ1 if halt else 0)

			# Train our network using target and predicted Q values
			_, summary = sess.run([updateModel, summary_op], feed_dict={inputs: [s], nextQ: targetQ})
			summary_writer.add_summary(summary, total_steps + steps)

			reward_sum += r

			s = s1

		# Reduce chance of random action as we train the model.
		# value = float(i+1)/num_episodes
		# e = epsilon(value)

		stat = dict()
		maxtile = max([max(game.currState.grid[k]) for k in range(len(game.currState.grid))])
		stat['maxTile'] = maxtile
		stat['score'] = game.currState.score
		stat['steps'] = steps
		stat['r'] = reward_sum
		stat['randsteps'] = "{0:.3f}".format(float(rand_steps)/steps)
		# stat['loss'] = l
		total_steps += steps

		add_scalar_summary({
			"steps": steps,
			"score": game.currState.score,
			"net_reward": reward_sum,
			"epsilon": e,
			"maxtile": maxtile,
			"threshold": tile_threshold * (2 ** i),
			"invalid_steps": invalid_steps,
			"last_500_maxtile_avg": 0 if len(maxtile_history) < 500 else np.mean(maxtile_history),
			"maxtile_avg_error": 0 if len(maxtile_history) < 500 else maxtile_history.count(tile_threshold * (2 ** i))/float(500),
			"exploration_ratio": float(stat['randsteps'])
		}, summary_writer,agent_id, game_i)

		print game_i, "\t", stat

		# update threshold if last 500 moves resulted in 15% error of threshold
		if len(maxtile_history) < 500:
			maxtile_history.append(maxtile)
		else:
			maxtile_history.pop(0)
			maxtile_history.append(maxtile)
			if maxtile_history.count(tile_threshold * (2 ** i))/float(500) >= 0.85:
				i += 1

		game_i += 1

	avg = float(score)/num_episodes
	print avg
	sess.close()

