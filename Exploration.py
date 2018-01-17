import numpy as np
import random

def softmax(action, allQ, i, epsilon, game):
	original_action = action[0]

	# Boltzman approach

	rand_action = False
	invalid_action = False

	logits = allQ/epsilon(i)
	logits = np.exp(logits)
	logits_sum = np.sum(logits)
	prob = logits/logits_sum

	invalid_action = []

	action[0] = np.random.choice([0, 1, 2, 3], p=prob[0])
	nextstate = game.transition(action[0])

	if not nextstate.halt:
		if not nextstate.valid:
			invalid_action = True
		while not nextstate.valid:
			invalid_action.append(action[0])
			while action[0] in invalid_action:
				action[0] = np.random.choice([0, 1, 2, 3], p=prob[0])
			nextstate = game.transition(action[0])

	if action[0] == original_action:
		rand_action = True

	return nextstate, action[0], rand_action, invalid_action


def egreedy(action, allQ, i, epsilon, game):

	random_action = False
	invalid_action = False
	policy_action = 0
	sorted_action = np.argsort(-np.array(allQ))[0]

	if np.random.rand(1) < epsilon(i):
		action[0] = random.randint(0, 3)
		random_action = True

	nextstate = game.transition(action[0])

	if not nextstate.halt:
		if not nextstate.valid and not random_action:
			invalid_action = True

		while not nextstate.valid:
			if random_action:
				b = action[0]
				while b == action[0]:
					b = random.randint(0, 3)
				action[0] = b
			else:  # ignore invalid action
				policy_action += 1
				action[0] = sorted_action[policy_action]
			nextstate = game.transition(action[0])


	return nextstate, action[0], random_action, invalid_action
