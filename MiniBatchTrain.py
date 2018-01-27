import ModelMgmt
import tensorflow as tf
import Game
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
update_mode = args["update-mode"]


# Initialize TF Network and variables
Qout, inputs = Network.getNetworkFromArgs(args["architecture"])
Qmean = tf.reduce_mean(Qout)
Qmax = tf.reduce_max(Qout)
predict = tf.argmax(Qout, 1)

# Initialize TF output and optimizer
nextQ = tf.placeholder(shape=[None, 4], dtype=tf.float32)
loss = Losses.getLossFromArgs(args["loss"])(nextQ, Qout)
trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

# Initialize tensorboard summary
summary_op = Summary.init_summary_writer(training_id=trainer_id, var_list=[("loss", loss), ("Qmean", Qmean), ("Qmax", Qmax)])


# Random action parameter
_epsilon = Gradients.Exponential(start=eps_start, stop=eps_stop)


def epsilon(i):
	# Exponentially decreasing epsilon to 0.1 for first 25% epochs, constant value of 0.1 from there on
	if i<num_episodes/(100.0/eps_steps):
		return _epsilon(i/(num_episodes/(100.0/eps_steps)))
	else:
		return eps_stop


memory = ReplayMemory(memory_size)


def update_model():
	if memory.full:
		replay = memory.sample(batch_size)
		state_list = []
		target_list = []

		for sample in replay:
			state = sample[0]
			action = sample[1]
			reward_list = sample[2]
			possible_states = sample[3]

			next_state = possible_states[action]

			_, allQ = sess.run([predict, Qout], feed_dict={inputs: [state]})
			Q1 = sess.run(Qout, feed_dict={inputs: [next_state]})

			# Obtain maxQ' and set our target value for chosen action.
			maxQ1 = np.max(Q1)
			targetQ = allQ

			targetQ[0, action] = reward_list[action] + gamma * maxQ1

			state_list.insert(0, state)
			target_list.insert(0, targetQ[0])

		_, summary = sess.run([updateModel, summary_op], feed_dict={inputs: state_list, nextQ: target_list})
		Summary.write_summary_operation(summary, total_steps + steps)


with tf.Session() as sess:
	sess.run(init)

	score = 0
	total_steps = 0

	for i in range(num_episodes):
		# Reset environment and get first new observation
		state = Game.new_game(4)
		s = normalize(state.grid_to_input())
		reward_sum = 0
		halt = False
		steps = 0
		rand_steps = 0
		invalid_steps = 0
		# The Q-Network
		while not halt:
			steps += 1
			if i == 0:
				state.printstate()
				print ""
			# Choose an action by greedily (with e chance of random action) from the Q-network
			a, allQ = sess.run([predict, Qout], feed_dict={inputs: [s]})

			possible_states, action, ra, invalid_prediction = exploration(a[0], allQ, i, epsilon, state)

			if ra:
				rand_steps += 1
			if invalid_prediction:
				invalid_steps += 1

			reward_list = []
			for k, nextstate in enumerate(possible_states):
				r = reward(state, nextstate)
				reward_list.insert(k, r)
				if r is not 0:
					r = np.log2(nextstate.score - state.score)/2.0

			reward_sum += reward_list[action]

			nextstate = possible_states[action]

			s1 = normalize(nextstate.grid_to_input())
			halt = nextstate.halt

			memory.push([s, action, reward_list, possible_states])

			# update step
			update_model()

			s = s1
			state = nextstate

		maxtile = max([max(state.grid[k]) for k in range(len(state.grid))])
		stat = {
			'max-tile': maxtile,
			'score': state.score,
			'steps': steps,
			'r': reward_sum,
			'rand-steps': "{0:.3f}".format(float(rand_steps) / steps)
		}
		total_steps += steps

		Summary.write_scalar_summaries([
			("steps", steps),
			("epsilon", epsilon(i)),
			("score", state.score),
			("rand-steps", float(rand_steps)/steps),
			("maxtile", maxtile),
			# ("invalid-steps", steps)
		], i)

		print i, "\t", stat

	sess.close()
