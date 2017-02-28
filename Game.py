import random


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


class Game:

	def __init__(self, size, grid=[]):
		self.grid = grid
		if not grid:
			self.emptygrid(size)

	def emptygrid(self, size, value=0):
		for i in range(size):
			self.grid.append([value] * size)

		self.add_random()
		self.add_random()

	def add_random(self):
		indices = []
		n = len(self.grid)
		for y in range(n):
			for x in range(n):
				if self.grid[y][x] == 0:
					indices.append((x, y))

		if len(indices) != 0:
			r = indices[random.randint(0, len(indices) - 1)]
			value = random.randint(1, 2) * 2
			self.grid[r[1]][r[0]] = value

	def acceptinput(self, direction=None, verbose=False):
		if direction:
			direction = getdirection(direction)
		else:
			direction = getkey()

		halt, valid, full = self.nextmove(direction, 0)

		if verbose:
			self.printgrid()
			print "Valid? ", valid, "\tHalt? ", halt
		return valid, full, halt

	def movelogic(self, direction, score):
		n = len(self.grid)

		if direction[0] != 0:
			d = direction[0]
		elif direction[1] != 0:
			d = direction[1]
			self.grid = map(list, zip(*self.grid))

		valid = False
		full = True

		for y in range(n):

			r = row = self.grid[y]
			r = filter(lambda a: a != 0, r)

			if d == 1:
				# Reverse
				r = r[::-1]

			x = 0

			while x < len(r) - 1:
				if r[x] == r[x + 1]:
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
				self.grid[y] = zeroes + r
			elif d == -1:
				self.grid[y] = r + zeroes

			if row != self.grid[y]:
				valid = True
				if len(filter(lambda a: a == 0, self.grid[y])):
					full = False

		if direction[1] != 0:
			self.grid = map(list, zip(*self.grid))

		return full, valid

	def nextmove(self, direction, score):

		n = len(self.grid)
		full, valid = self.movelogic(direction, score)

		if not full and valid:
			self.add_random()

		halt = False
		if full:  # Check for game end
			halt = True
			for i in range(n):
				for j in range(n - 1):
					if self.grid[j][i] == self.grid[j + 1][i] or self.grid[i][j] == self.grid[i][j + 1]:
						halt = False
						break

		return halt, valid, full

	def grid_to_input(self):
		arr = []
		for i in range(len(self.grid)):
			arr = arr + self.grid[i]
		return arr

	def printgrid(self):
		for i in range(len(self.grid)):
			print self.grid[i]


default = [
	[0, 0, 0, 0],
	[4, 4, 4, 0],
	[0, 0, 0, 0],
	[0, 0, 0, 0]
]

game = Game(4)
game.printgrid()

print game.acceptinput(direction=[ 0.99990205,  0.99985169,  0.99952689,  0.99978909])
game.printgrid()