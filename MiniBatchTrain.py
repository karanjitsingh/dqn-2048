import ModelMgmt
import tensorflow as tf
from Game import Game
from functions import Gradients
from functions.General import *
import Network
import Summary
import Exploration
import Losses

args = ModelMgmt.parse_cli()

#  Set learning parameters
gamma = args["discount-factor"]
num_episodes = args["epochs"]
learning_rate = args["learning-rate"]
[eps_start, eps_stop, eps_steps] = args["epsilon-params"]
trainer_id = args["trainer-id"]
memory_size = args["replay-size"]
exploration = Exploration.getExplorationFromArgs(args["exploration"])
batch_size = args["batch-size"]

# Random action parameter
_epsilon = Gradients.Exponential(start=eps_start, stop=eps_stop)


def epsilon(i):
	# Exponentially decreasing epsilon to 0.1 for first 25% epochs, constant value of 0.1 from there on
	if i<num_episodes/(100.0/eps_steps):
		return _epsilon(i/(num_episodes/(100.0/eps_steps)))
	else:
		return eps_stop


# Initiailize TF Network and variables
Qout, inputs = Network.getNetworkFromArgs(args["architecture"])
Qmean = tf.reduce_mean(Qout)
Qmax = tf.reduce_max(Qout)
predict = tf.argmax(Qout, 1)


nextQ = tf.placeholder(shape=[None, 4], dtype=tf.float32)
loss = Losses.getLossFromArgs(args["loss"])(nextQ, Qout)
trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

summary_op = Summary.init_summary_writer(training_id=trainer_id, var_list=[("loss", loss), ("Qmean", Qmean), ("Qmax", Qmax)])

memory = ReplayMemory(memory_size)

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
				print ""
			# Choose an action by greedily (with e chance of random action) from the Q-network
			a, allQ = sess.run([predict, Qout], feed_dict={inputs: [s]})

			nextstate, a[0], ra, ia = exploration(a, allQ, i, epsilon, game)

			# Get new state and reward from environment

			r = reward(currstate, nextstate)
			maxtile = max([max(game.currState.grid[k]) for k in range(len(game.currState.grid))])
			if r is not 0:
				r = np.log2(nextstate.score - currstate.score)/2.0
			reward_sum += r

			s1 = normalize(game.grid_to_input())
			halt = nextstate.halt

			memory.push([s, a[0], r, s1])

			# Feed-forward
			if memory.full:
				replay = memory.sample(batch_size)
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
				Summary.write_summary_operation(summary, total_steps + steps)

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

		Summary.write_scalar_summaries([
			("steps", steps),
			("epsilon", epsilon(i)),
			("score", game.currState.score),
			("rand-steps", float(rand_steps)/steps),
			("maxtile", maxtile),
			# ("invalid-steps", steps)
		], i)

		print i, "\t", stat

	sess.close()
