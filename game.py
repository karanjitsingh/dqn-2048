import random

def add_random(grid):
	indices = []
	n = len(grid)
	for y in range(n):
		for x in range(n):
			if(grid[y][x] == 0 ):
				indices.append((x,y))

	if len(indices)!=0:
		r = indices[random.randint(0,len(indices)-1)]
		value = random.randint(1,2)*2
		grid[r[1]][r[0]]= value
		return grid,False
	else:
		return grid,True


def new_grid(size,value=0):

	grid = []
	for i in range(size):
		grid.append([value] * size)

	grid,blank = add_random(grid)
	grid,blank = add_random(grid)


	return grid


def move(grid, direction, score):
	halt = False

	n = len(grid)

	if direction[0]!=0:
		d=direction[0]
	elif direction[1]!=0:
		d=direction[1]
		grid = map(list,zip(*grid))

	if d==1:
		for y in range(n):
			for x in range(n-1,0,-1):
				row = grid[y]
				if row[x] == row[x-1] and row[x]!=0:
					row[x] = row[x]*2
					row[x-1] = 0
			row = filter(lambda a:a!=0,row)
			row = [0]*(len(grid[y]) - len(row)) + row
			grid[y] = row
	elif d == -1:
		for y in range(n):
			for x in range(0,n-1,1):
				row = grid[y]
				if row[x] == row[x+1] and row[x]!=0:
					row[x] = row[x]*2
					row[x+1] = 0
			row = filter(lambda a:a!=0,row)
			row = row+[0]*(len(grid[y]) - len(row))
			grid[y] = row

	if direction[1]!=0:
		d=direction[1]
		grid = map(list,zip(*grid))
	#if direction[1]==1:
	#	for x in range(n):



	grid,halt = add_random(grid)

	if halt==True:
		print "Game Over"

	for y in range(n):
		print grid[y]
	return grid,halt


def grid_to_input(grid):
	arr = []
	for i in range(len(grid)):
		arr = arr + grid[i]
	return arr

grid = new_grid(4)