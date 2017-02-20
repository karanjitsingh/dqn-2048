import random

def getKey():
	while(1):
		k = str(raw_input("Next: "));
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
	indices = [];
	n = len(grid);
	for y in range(n):
		for x in range(n):
			if(grid[y][x] == 0 ):
				indices.append((x,y))

	if len(indices)!=0:
		r = indices[random.randint(0,len(indices)-1)]
		value = random.randint(1,2)*2
		grid[r[1]][r[0]]= value
		return grid;


def emptyGrid(size,value=0):
	grid = []
	for i in range(size):
		grid.append([value] * size)

	grid = add_random(grid)
	grid = add_random(grid)

	return grid

def fullGrid(size):
	grid= [];
	for i in range(size):
		grid.append([size * i + j + 1 for j in range(size)]);
	return grid;


def moveLogic(grid,direction,score):
	n = len(grid);

	if direction[0]!=0:
		d=direction[0]
	elif direction[1]!=0:
		d=direction[1]
		grid = map(list,zip(*grid))

	valid = False;
	full = True;

	for y in range(n):
		r = row = grid[y];
		r = filter(lambda a:a!=0, r);
		x = 0;
		
		while x<len(r)-1:
			if(r[x] == r[x+1]):
				r[x] = reduce(lambda x,y: x+y,r[x:x+2]);
				r[x+1] = 0;
				x=x+1;
			x=x+1;

		r = filter(lambda a:a!=0, r);
		zeroes = [0 for i in range(n-len(r))];
		if(d==1):
			grid[y] =  zeroes + r;
		elif(d==-1):
			grid[y] = r + zeroes;

		if(row != grid[y]):
			valid=True;
			if(len(filter(lambda a:a==0, grid[y]))):
				full=False;



	if direction[1]!=0:
		d=direction[1];
		grid = map(list,zip(*grid));

	return grid,full,valid;



def nextMove(grid, direction, score):
	halt = False
	n = len(grid);

	grid,full,valid = moveLogic(grid,direction,score);
	
	print "f", full;

	if(not full and valid):
		grid = add_random(grid);

	halt = False;
	if(full):		# Check for game end
		halt = True;
		for i in range(n):
			for j in range(n-1):
				if(grid[j][i] == grid[j+1][i] or grid[i][j] == grid[i][j+1]):
					halt = False;
					break;


	return grid,halt,valid,full;


def grid_to_input(grid):
	arr = []
	for i in range(len(grid)):
		arr = arr + grid[i]
	return arr

def printGrid(grid):
	for i in range(len(grid)):
		print grid[i];

grid = emptyGrid(4);
printGrid(grid);

while(1):
	direction = getKey();
	if(direction):
		grid,halt,valid,_ = nextMove(grid,direction,0);
		printGrid(grid);
		print "Valid? ", valid, "\tHalt? ", halt;