import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import random
import tensorflow as tf
from Game import Game
from functions import Gradients
from functions.General import *
import math
import argparse
import ast

#
# Process command line args
#
argv = [sys.argv[0], "[16,256,4]", "0.1", "0.8", "50000"]
print argv
sys.argv = argv

args = []
hidden_layers = []

trainer_id = 'AAA'

def parse_cli():
	parser = argparse.ArgumentParser()
	parser.add_argument("structure", help="Network structure python array")
	parser.add_argument("learning-rate", help="Learning rate of model")
	parser.add_argument("discount-factor", help="Discount factor of model")
	parser.add_argument("epochs", help="Number of games to run")

	global args
	global hidden_layers
	args = vars(parser.parse_args())
	for key, value in args.iteritems():
		args[key] = ast.literal_eval(value)

	hidden_layers = args["structure"][1:-1]

parse_cli()
#
# End
#


#
# Init tensorflow variables
#
tf.reset_default_graph()

inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)


def classic_nn():
	last_layer = 16
	weights = []
	biases = []

	activation = [inputs1]

	for i, n in enumerate(hidden_layers):
		weights.append(tf.Variable(tf.truncated_normal([last_layer, n], 0, 0.1)))
		biases.append(tf.Variable(tf.constant(0.1, shape=[n])))
		activation.append(tf.nn.relu(tf.add(tf.matmul(activation[i], weights[i]), biases[i])))
		last_layer = n

	weights.append(tf.Variable(tf.truncated_normal([last_layer, 4], 0, 0.1)))
	biases.append(tf.Variable(tf.constant(0.1, shape=[4])))

	return tf.nn.relu(tf.add(tf.matmul(activation[-1], weights[-1]), biases[-1]))


def conv_nn():
	def new_weights(shape):
		return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


	def new_biases(length):
		return tf.Variable(tf.constant(0.1, shape=[length]))


	def new_conf_layer(input, filter_size, num_filters):
		shape = [filter_size, filter_size,1,num_filters]
		weights = new_weights(shape)
		biases = new_biases(length=num_filters)

		layer=tf.nn.conv2d(input=input,
						   filter=weights,
						   strides=[1,1,1,1],
						   padding='SAME') + biases

		layer = tf.nn.relu(layer)

		return layer, weights

	def new_fc_layer(input,
					 num_inputs,
					 num_outputs,
					 use_relu=True):

		weights = new_weights(shape=[num_inputs, num_outputs])
		biases = new_biases(length=num_outputs)

		layer = tf.matmul(input, weights) + biases

		if use_relu:
			layer = tf.nn.relu(layer)

		return layer

	def flatten_layer(layer):

		layer_shape = layer.get_shape()
		num_features = np.array(layer_shape[1:4], dtype=int).prod()
		layer_flat = tf.reshape(layer, [-1, num_features])

		return layer_flat, num_features

	inputs_2d= tf.reshape(inputs1, [-1, 4, 4, 1])

	layer_1, _ = new_conf_layer(inputs_2d, filter_size=2, num_filters=9)



	layer_flat, num_features = flatten_layer(layer_1)


	layer_fc1 = new_fc_layer(input = layer_flat,
							 num_inputs= num_features,
							 num_outputs= 64)

	layer_fc2 = new_fc_layer(input= layer_fc1,
							 num_inputs= 64,
							 num_outputs= 4)

	return layer_fc2


def write_scalar_summaries(summary_writer, summary_id, step, summary_list):
	for t in summary_list:
		s = tf.Summary(value=[
			tf.Summary.Value(tag=str(summary_id + "/" + t[0]), simple_value=t[1]),
		])

		summary_writer.add_summary(s, step)



# Qout = classic_nn()
Qout = conv_nn()
Qmean = tf.reduce_mean(Qout)
Qmax = tf.reduce_max(Qout)
predict = tf.argmax(Qout, 1)


# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=args["learning-rate"])
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

#  Set learning parameters
y = args["discount-factor"]
e = 0.15

epsilon = Gradients.Exponential(start=0.9, stop=0.001)
num_episodes = args["epochs"]


with tf.name_scope(trainer_id):
	tf.summary.scalar("loss", loss)
	tf.summary.scalar("Qmean", Qmean)

summary_op = tf.summary.merge_all()
#
# End
#


# create lists to contain total rewards and steps per episode
rList = []
with tf.Session() as sess:
	sess.run(init)

	writer = tf.summary.FileWriter("./log", graph=tf.get_default_graph())
	score = 0
	total_steps = 0

	x2 = ""
	for i in range(num_episodes):
		# Reset environment and get first new observation
		game = Game(4)
		s = normalize(game.grid_to_input())
		reward_sum = 0
		halt = False
		steps = 0
		rand_steps = 0
		# The Q-Network
		while not halt:
			steps += 1
			currstate = game.currState
			if i == 0:
				game.printgrid()
			# Choose an action by greedily (with e chance of random action) from the Q-network
			a, allQ = sess.run([predict, Qout], feed_dict={inputs1: [s]})

			random_action = False
			policy_action = 0
			sorted_action = np.argsort(-np.array(allQ))[0]

			if np.random.rand(1) < (epsilon(i/15000.0) if i < 15000 else 0):
				a[0] = random.randint(0, 3)
				rand_steps += 1
				random_action = True
				if i == 0:
					print "random action: ", a[0]
					print ""
			elif i == 0:
				print "policy action: ", a[0]
				print ""

			nextstate = game.transition(a[0])

			if not nextstate.halt and not nextstate.valid:
				if random_action:
					b = a[0]
					while b == a[0]:
						b = random.randint(0, 3)
					a[0] = b
				# else:
				# 	policy_action += 1
				# 	a[0] = sorted_action[policy_action]
				nextstate = game.transition(a[0])

			# Get new state and reward from environment

			maxtile = max([max(game.currState.grid[k]) for k in range(len(game.currState.grid))])
			r = reward(currstate, nextstate) * np.log2(maxtile) / 10.0

			s1 = normalize(game.grid_to_input())
			halt = nextstate.halt
			#
			# if halt:
			# 	r = -1

			# Obtain the Q' values by feeding the new state through our network
			Q1 = sess.run(Qout, feed_dict={inputs1: [s1]})
			# Obtain maxQ' and set our target value for chosen action.
			maxQ1 = np.max(Q1)
			targetQ = allQ
			targetQ[0, a[0]] = r + (0 if halt else y*maxQ1)

			# Train our network using target and predicted Q values
			_, summary = sess.run([updateModel, summary_op], feed_dict={inputs1: [s], nextQ: targetQ})
			writer.add_summary(summary, total_steps + steps)

			reward_sum += r

			s = s1

		stat = dict()
		maxtile = max([max(game.currState.grid[k]) for k in range(len(game.currState.grid))])
		stat['maxTile'] = maxtile
		stat['score'] = game.currState.score
		stat['steps'] = steps
		stat['r'] = reward_sum
		stat['randsteps'] = "{0:.3f}".format(float(rand_steps)/steps)
		# stat['loss'] = l
		total_steps += steps

		write_scalar_summaries(writer, trainer_id, i, [
			("steps", steps),
			("epsilon", (epsilon(i/15000.0) if i < 15000 else 0)),
			("score", game.currState.score),
			("net_reward", reward_sum),
			("rand_steps", float(rand_steps)/steps),
			("maxtile", maxtile)
		])

		print i, "\t", stat
		rList.append(reward_sum)

	avg = float(score)/num_episodes
	print avg
	sess.close()

# print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"
