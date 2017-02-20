import random

def getKey():
	k = str(raw_input("Direction: "));
	while(1):
		if(k=='w' or k=='W'):
			return getDirection('up');
		elif k=='a' or k=='A':
			return getDirection('left');
		elif k=='s' or k=='S':
			return getDirection('down');
		elif k=='d' or k=='D':
			return getDirection('right');

def getDirection(x):
	return {
		'up': [0,-1],
		'down': [0,1],
		'right': [1,0],
		'left': [-1,0]
	}[x];

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

def nextMove(grid, direction, score):
	halt = False

	n = len(grid)

	if direction[0]!=0:
		d=direction[0]
	elif direction[1]!=0:
		d=direction[1]
		grid = map(list,zip(*grid))


	for y in range(n):
		r = grid[y];
		r = filter(lambda a:a!=0, r);
		x = 0;
		
		while x<len(r)-1:
			print x;
			if(r[x] == r[x+1]):
				r[x] = reduce(lambda x,y: x+y,r[x:x+2]);
				r[x+1] = 0;
				x=x+1;
			x=x+1;

		r = filter(lambda a:a!=0, r);
		row = [0 for i in range(n-len(r))];
		if(d==1):
			grid[y] =  row + r;
		elif(d==-1):
			grid[y] = r + row;



	if direction[1]!=0:
		d=direction[1]
		grid = map(list,zip(*grid))


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

def printGrid(grid):
	for i in range(len(grid)):
		print grid[i];
	print " yo";

grid = new_grid(4);
printGrid(grid);


while(1):
	direction = getKey();
	if(direction):
		grid,halt = nextMove(grid,direction,0);
