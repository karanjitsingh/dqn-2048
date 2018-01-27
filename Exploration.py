import numpy as np
import random


def getAllStates(game):
	state_list = []
	invalid_list = []
	for i in range(4):
		state = game.transition(i)
		state_list.insert(i, state)
		invalid_list.insert(i, not state.valid)

	return state_list, invalid_list


def softmax(action, allQ, i, epsilon, game):
	original_action = action[0]

	# Boltzman approach

	rand_action = False

	logits = allQ/epsilon(i)
	logits = np.exp(logits)
	logits_sum = np.sum(logits)
	prob = logits/logits_sum

	action[0] = np.random.choice([0, 1, 2, 3], p=prob[0])
	state_list, invalid_list = getAllStates(game)
	nextstate = state_list[action[0]]

	if not nextstate.halt:
		while not nextstate.valid:
			while invalid_list[action[0]]:
				action[0] = np.random.choice([0, 1, 2, 3], p=prob[0])
			nextstate = state_list[action[0]]

	if action[0] != original_action:
		rand_action = True

	return nextstate, action[0], rand_action, invalid_list


def egreedy(action, allQ, i, epsilon, game):

	random_action = False
	policy_action = 0
	sorted_action = np.argsort(-np.array(allQ))[0]

	if np.random.rand(1) < epsilon(i):
		action[0] = random.randint(0, 3)
		random_action = True

	state_list, invalid_list = getAllStates(game)
	nextstate = state_list[action[0]]

	if not nextstate.halt:
		while not nextstate.valid:
			if random_action:
				b = action[0]
				while b == action[0]:
					b = random.randint(0, 3)
				action[0] = b
			else:  # ignore invalid action
				policy_action += 1
				action[0] = sorted_action[policy_action]

			nextstate = state_list[action[0]]

	return state_list, action[0], random_action, invalid_list


def getExplorationFromArgs(args):
	if args == "egreedy":
		return egreedy
	if args == "softmax":
		return softmax
