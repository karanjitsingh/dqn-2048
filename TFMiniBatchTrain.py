import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import random
import tensorflow as tf
from Game import Game
from functions import Gradients
from functions.General import *
import argparse
import ast
import TFNetwork
import TFSummary
import TFLosses


default_args = [sys.argv[0], "[16,256,256,256,4]", "0.1", "0.8", "30000", "\"AAA\""]
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


args, hidden_layers, trainer_id = parse_cli()


# Initiailize TF Network and variables
Qout, inputs = TFNetwork.new_FCN(16, hidden_layers, 4)
Qmean = tf.reduce_mean(Qout)
Qmax = tf.reduce_max(Qout)
predict = tf.argmax(Qout, 1)


nextQ = tf.placeholder(shape=[None, 4], dtype=tf.float32)
loss = tf.losses.huber_loss(nextQ, Qout)
trainer = tf.train.GradientDescentOptimizer(learning_rate=args["learning-rate"])
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

#  Set learning parameters
gamma = args["discount-factor"]
num_episodes = args["epochs"]

# Random action strategy
_epsilon = Gradients.Exponential(start=1, stop=0.1)


def epsilon(i):
	# Exponentially decreasing epsilon to 0.1 for first 25% epochs, constant value of 0.1 from there on
	if i<num_episodes/4.0:
		return _epsilon(i/(num_episodes/4.0))
	else:
		return 0.1


summary_op = TFSummary.init_summary_writer(training_id=trainer_id, var_list=[("loss", loss), ("Qmean", Qmean), ("Qmax", Qmax)])


memory = ReplayMemory(10000)


with tf.Session() as sess:
	sess.run(init)

	score = 0
	total_steps = 0

	for i in range(num_episodes):
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
			if i == 0:
				game.printgrid()
			# Choose an action by greedily (with e chance of random action) from the Q-network
			a, allQ = sess.run([predict, Qout], feed_dict={inputs: [s]})



			# Boltzman approach

			logits = allQ/((1 - i/float(num_episodes)) * 1000.0)
			logits = np.exp(logits)
			logits_sum = np.sum(logits)
			prob = logits/logits_sum

			invalid_action=[]
			a[0] = np.random.choice([0,1,2,3], p=prob)
			nextstate = game.transition(a[0])

			if not next_state.halt:
				while not next_state.valid:
					invalid_action.append(a[0])
					while a[0] in invalid_action:
						a[0] = np.random.choice([0, 1, 2, 3], p=prob)
					nextstate = game.transition(a[0])

			# e-greedy approach
			#
			# random_action = False
			# policy_action = 0
			# sorted_action = np.argsort(-np.array(allQ))[0]
			#
			# if np.random.rand(1) < epsilon(i):
			# 	a[0] = random.randint(0, 3)
			# 	rand_steps += 1
			# 	random_action = True
			# 	if i == 0:
			# 		print "random action: ", a[0]
			# 		print ""
			# elif i == 0:
			# 	print "policy action: ", a[0]
			# 	print ""
			#
			# nextstate = game.transition(a[0])
			#
			# if not nextstate.halt:
			# 	if not nextstate.valid and not random_action:
			# 		invalid_steps += 1
			#
			# 	while not nextstate.valid:
			# 		if random_action:
			# 			b = a[0]
			# 			while b == a[0]:
			# 				b = random.randint(0, 3)
			# 			a[0] = b
			# 		else:  # ignore invalid action
			# 			policy_action += 1
			# 			a[0] = sorted_action[policy_action]
			# 		nextstate = game.transition(a[0])



			# Get new state and reward from environment

			r = reward(currstate, nextstate)
			maxtile = max([max(game.currState.grid[k]) for k in range(len(game.currState.grid))])
			if r is not 0:
				# r = np.log2(nextstate.score - currstate.score)/10.0
				r = np.log2(maxtile)/10.0
			reward_sum += r

			s1 = normalize(game.grid_to_input())
			halt = nextstate.halt

			memory.push([s, a[0], r, s1])

			# Feed-forward
			if memory.full:
				replay = memory.sample(15)
				state_list = []
				target_list = []

				for sample in replay:
					state = sample[0]
					action = sample[1]
					rr = sample[2]
					next_state = sample[3]

					_, allQ = sess.run([predict, Qout], feed_dict={inputs: [state]})
					Q1 = sess.run(Qout, feed_dict={inputs: [next_state]})

					# Obtain maxQ' and set our target value for chosen action.
					maxQ1 = np.max(Q1)
					targetQ = allQ
					targetQ[0, action] = rr + gamma*maxQ1

					state_list.insert(0, state)
					target_list.insert(0, targetQ[0])

				_, summary = sess.run([updateModel, summary_op], feed_dict={inputs: state_list, nextQ: target_list})
				TFSummary.write_summary_operation(summary, total_steps + steps)

			s = s1

		maxtile = max([max(game.currState.grid[k]) for k in range(len(game.currState.grid))])
		stat = {
			'max-tile': maxtile,
			'score': game.currState.score,
			'steps': steps,
			'r': reward_sum,
			'rand-steps': "{0:.3f}".format(float(rand_steps) / steps)
		}
		total_steps += steps

		TFSummary.write_scalar_summaries([
			("steps", steps),
			("epsilon", epsilon(i)),
			("score", game.currState.score),
			("rand-steps", float(rand_steps)/steps),
			("maxtile", maxtile),
			("invalid-steps", invalid_steps)
		], i)

		print i, "\t", stat

	sess.close()
