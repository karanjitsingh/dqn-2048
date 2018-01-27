import random
import copy


class State(object):
	def __init__(self, grid, score, empty_tiles, halt=False, full=False, valid=True):
		self.grid = grid
		self.score = score
		self.halt = halt
		self.full = full
		self.valid = valid
		self.empty_tiles = empty_tiles

	def printstate(self):
		grid = self.grid
		for i in range(len(grid)):
			print grid[i]
		print (self.score, self.halt, self.full, self.valid)

	def grid_to_input(self):
		arr = []
		grid = self.grid
		for i in range(len(grid)):
			arr = arr + grid[i]
		return arr


def getkey():
	while 1:
		k = str(raw_input("Next: "))
		if k == 'w' or k == 'W':
			return getdirection('up')
		elif k == 'a' or k == 'A':
			return getdirection('left')
		elif k == 's' or k == 'S':
			return getdirection('down')
		elif k == 'd' or k == 'D':
			return getdirection('right')


def getdirection(x):
	directions = {
		'up': [0, -1],
		'down': [0, 1],
		'right': [1, 0],
		'left': [-1, 0]
	}
	if isinstance(x, str):
		return directions[x]
	elif isinstance(x, int):
		return directions.values()[x]
	elif isinstance(x, list):
		return directions.values()[x.index(max(x))]


def get_next_state(state, direction):
	grid = copy.copy(state.grid)
	n = len(grid)
	score = 0

	if direction[0] != 0:
		d = direction[0]
	elif direction[1] != 0:
		d = direction[1]
		grid = map(list, zip(*grid))

	valid = False
	full = True

	empty_tiles = 0

	for y in range(n):
		r = row = grid[y]
		r = filter(lambda a: a != 0, r)
		if d == 1:
			# Reverse
			r = r[::-1]

		x = 0
		while x < len(r) - 1:
			# Merge
			if r[x] == r[x + 1]:
				score += r[x]*2
				r[x] = reduce(lambda x, y: x + y, r[x:x + 2])
				r[x + 1] = 0
				x += 1
			x += 1

		r = filter(lambda a: a != 0, r)
		if d == 1:
			# Reverse
			r = r[::-1]
		zeroes = [0] * (n - len(r))
		if d == 1:
			grid[y] = zeroes + r
		elif d == -1:
			grid[y] = r + zeroes
		if row != grid[y]:
			valid = True

		empty_tiles += len(zeroes)

	if empty_tiles > 1:
		full = False

	if empty_tiles > 0 and valid:
		grid = add_random(grid)
		empty_tiles -= 1

	if direction[1] != 0:
		grid = map(list, zip(*grid))

	halt = False
	if full:  # Check for game end
		halt = True
		for i in range(n):
			for j in range(n - 1):
				if grid[j][i] == 0 or grid[j][i] == grid[j + 1][i] or grid[i][j] == grid[i][j + 1]:
					halt = False
					break

	return State(grid, state.score+score, empty_tiles, halt, full, valid)


def add_random(grid):
	indices = []
	n = len(grid)
	for y in range(n):
		for x in range(n):
			if grid[y][x] == 0:
				indices.append((x, y))

	if len(indices) != 0:
		r = indices[random.randint(0, len(indices) - 1)]
		value = random.randint(1, 2) * 2
		grid[r[1]][r[0]] = value

	return grid


def new_game(size, grid=[]):
	if not grid:
		for i in range(size):
			grid.append([0] * size)

		grid = add_random(add_random(grid))

	return State(grid, 0, size*size - 2)


def transition(state, direction=None):
	if direction is not None:
		direction = getdirection(direction)
	else:
		direction = getkey()

	state = get_next_state(state, direction)
	return state
