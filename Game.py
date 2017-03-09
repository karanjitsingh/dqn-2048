import random
import copy


class State(object):
	def __init__(self, grid, score, halt=False, full=False, valid=True):
		self.grid = grid
		self.score = score
		self.halt = halt
		self.full = full
		self.valid = valid

	def printstate(self):
		grid = self.grid
		for i in range(len(grid)):
			print grid[i]
		print (self.score, self.halt, self.full, self.valid)


class Game(object):

	@staticmethod
	def getkey():
		while 1:
			k = str(raw_input("Next: "))
			if k == 'w' or k == 'W':
				return Game.getdirection('up')
			elif k == 'a' or k == 'A':
				return Game.getdirection('left')
			elif k == 's' or k == 'S':
				return Game.getdirection('down')
			elif k == 'd' or k == 'D':
				return Game.getdirection('right')

	@staticmethod
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

	@staticmethod
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
				if len(filter(lambda a: a == 0, grid[y])):
					full = False

		if direction[1] != 0:
			grid = map(list, zip(*grid))

		halt = False
		if full:  # Check for game end
			halt = True
			for i in range(n):
				for j in range(n - 1):
					if grid[j][i] == grid[j + 1][i] or grid[i][j] == grid[i][j + 1]:
						halt = False
						break

		if not full and valid:
			grid = Game.add_random(grid)

		return State(grid, state.score+score, halt, full, valid)

	@staticmethod
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

	def __init__(self, size, grid=[]):
		self.grid = grid
		if not grid:
			self.emptygrid(size)
		self.initState = State(grid, 0)
		self.currState = self.initState

	def emptygrid(self, size, value=0):
		for i in range(size):
			self.grid.append([value] * size)

		self.grid = Game.add_random(Game.add_random(self.grid))

	def transition(self, direction=None):
		if direction is not None:
			direction = Game.getdirection(direction)
		else:
			direction = Game.getkey()
		self.currState = Game.get_next_state(self.currState, direction)
		return self.currState

	def grid_to_input(self):
		arr = []
		grid = self.currState.grid
		for i in range(len(grid)):
			arr = arr + grid[i]
		return arr

	def printgrid(self):
		grid = self.currState.grid
		for i in range(len(grid)):
			print grid[i]


# default = [
# 	[0, 0, 0, 0],
# 	[4, 4, 4, 0],
# 	[0, 0, 0, 0],
# 	[0, 0, 0, 0]
# ]
#
# game = Game(4)
# game.printgrid()
#
# while 1:
# 	state = game.transition()
# 	game.printgrid()
# 	print state.valid, state.halt, state.full
