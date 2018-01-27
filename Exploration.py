import numpy as np
import random
import Game


def getAllStates(state):
	state_list = []
	invalid_list = []
	for i in range(4):
		state = Game.transition(state, i)
		state_list.insert(i, state)
		invalid_list.insert(i, not state.valid)

	return state_list, invalid_list


def softmax(action, allQ, i, epsilon, state):
	original_action = action

	# Boltzman approach

	rand_action = False

	logits = allQ/epsilon(i)
	logits = np.exp(logits)
	logits_sum = np.sum(logits)
	prob = logits/logits_sum

	action = np.random.choice([0, 1, 2, 3], p=prob[0])
	state_list, invalid_list = getAllStates(state)
	nextstate = state_list[action]

	if not nextstate.halt:
		while not nextstate.valid:
			while invalid_list[action]:
				action = np.random.choice([0, 1, 2, 3], p=prob[0])
			nextstate = state_list[action]

	if action != original_action:
		rand_action = True

	return nextstate, action, rand_action, invalid_list[action]


def egreedy(action, allQ, i, epsilon, state):

	random_action = False
	policy_action = 0
	sorted_action = np.argsort(-np.array(allQ))[0]

	if np.random.rand(1) < epsilon(i):
		action = random.randint(0, 3)
		random_action = True

	state_list, invalid_list = getAllStates(state)
	nextstate = state_list[action]

	if not nextstate.halt:
		while not nextstate.valid:
			if random_action:
				b = action
				while b == action:
					b = random.randint(0, 3)
				action = b
			else:  # ignore invalid action
				policy_action += 1
				action = sorted_action[policy_action]

			nextstate = state_list[action]

	return state_list, action, random_action, invalid_list


def getExplorationFromArgs(args):
	if args == "egreedy":
		return egreedy
	if args == "softmax":
		return softmax
