import numpy as np
from queue import PriorityQueue
import matplotlib.pylab as plt
import math
import random

class mazeCell(object):
    def __init__(self, x, y, open):        
        self.open = open      #True if accessible, False if blocked
        self.x = x            #x co-ordinate of the cell
        self.y = y            #y co-ordinate of the cell
        self.parent = None
        self.h = 0
        self.f = 0
    

class mazeAstarSolver(object):
    def __init__(self, maze):
        self.fringe = PriorityQueue()    
        self.visited = set()   #keeps track of visited cells
        self.cost = {}         #dict containing all the cells considered along with updated cost
        self.cells = []        #list of all cells
        self.dim = 20
        self.max_frig_size = 0
        self.total_nodes = 0
        self.finish = False
        self.maze = maze    
        
    def init_maze(self):        
        self.maze = np.reshape(self.maze, (self.dim, self.dim))     #convert 1D numpy array to 2D array with dim of the maze
        for x in range(self.dim):
            for y in range(self.dim):
                if self.maze[x][y] == 0:
                    open = False
                else:
                    open = True
                self.cells.append(mazeCell(x, y, open)) #adding all cells to the 'cells' list
        self.start = self.get_cell(0, 0)
        self.end = self.get_cell(self.dim - 1, self.dim - 1)
        
    def get_h(self, cell, heu):
        if heu == "e":
            #cost of 10 is given by default to move in any direction
            #euclidean distance
            return 10 * (math.sqrt(math.pow(cell.x - self.end.x,2) + math.pow(cell.y - self.end.y,2)))
        else:
            #manhattan distance
            return 10 * (abs(cell.x - self.end.x) + abs(cell.y - self.end.y))
    
    def get_cell(self, x, y):      #returns the specified cell from the 'cells' list
        return self.cells[x * self.dim + y]
    
    def get_adjacent_cells(self, cell):
        #returns all possible adjacent cells of the specified cell
        cells = []
        if cell.x < self.dim-1:
            cells.append(self.get_cell(cell.x+1, cell.y))
        if cell.y > 0:
            cells.append(self.get_cell(cell.x, cell.y-1))
        if cell.x > 0:
            cells.append(self.get_cell(cell.x-1, cell.y))
        if cell.y < self.dim-1:
            cells.append(self.get_cell(cell.x, cell.y+1))
        return cells
          
    def solve(self):  
		#start and end cells are added only to check whether they're solvable and to calculate their hardness score
		#because otherwise they may get mutated in the reproduction process
        self.maze = np.append(np.array([1]),self.maze)    #adding 1 (start node) to the maze arrangement
        self.maze = np.append(self.maze, np.array([1]))   #adding 1 (end node) to the maze arrangement
        self.init_maze()
        heu = 'm'                                       #default heuristic set to Manhattan distance
        self.fringe.put((self.start.f, 0, self.start))  #adding start cell to the fringe
        self.max_frig_size = self.fringe.qsize()
        self.cost[self.start] = 0
        i = 1
        while not self.fringe.empty():
            f, x, cell = self.fringe.get()
            self.visited.add(cell)       #add cells to visited set as and when they're visited
            if cell is self.end:         #if popped cell is target cell, then path has been found
                self.finish = True
                self.total_nodes = i     #no. of nodes expanded
                return self.finish, self.max_frig_size, self.total_nodes   #returns whether the maze is solvable or not, maximal fringe size
																		   #and the maximum no. of nodes expanded respectively
            adj_cells = self.get_adjacent_cells(cell)
            for adj_cell in adj_cells:
                if adj_cell.open and adj_cell not in self.visited:
                    c = self.cost[cell] + 10   #adding g score
                    if adj_cell not in self.cost or self.cost[adj_cell] > c:
                    #if cell hasn't been seen before or has worse cost than the new cost
                        self.cost[adj_cell] = c;          #update to new cost
                        adj_cell.parent = cell         
                        f = c + self.get_h(adj_cell, heu) #get f value
                        self.fringe.put((f,i,adj_cell))   #push in fringe with f as priority
                        if(self.fringe.qsize() > self.max_frig_size):
                            self.max_frig_size = self.fringe.qsize()
                        i += 1
      
        self.total_nodes = i     
                   
        return self.finish, self.max_frig_size, self.total_nodes

def remove_from_npa(L,arr):       #function to remove a np array from a list of np arrays
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

#function to generate the contents (excluding start and end cells) of the maze of dim 20
def generate_maze():              
    CELL_BLOCKED=0
    CELL_OPEN=1        
    dim = 20
    maze = np.random.choice([CELL_BLOCKED, CELL_OPEN], (dim * dim - 2), p=[0.2,0.8])
    #maze[0][0] = CELL_PATH
    #maze[dim-1][dim-1] = CELL_PATH
    return maze
 
#selecting the fittest(maze with the max hardness score) in the chosen generation 
def selection(population, mode):        
    fittest = (np.zeros(398, dtype = 'int'))
    hardness = -100
    for i in population:
        p = mazeAstarSolver(i)
        if mode == 'f':                 #f - max fringe size as hardness score
            h_score = p.solve()[1]
        else:							#n - max no. of nodes expanded as hardness score
            h_score = p.solve()[2]
        if h_score > hardness:
            hardness = h_score
            fittest = i
    
    return fittest, hardness            #return the fittest maze with corresponding hardness score
    
#function that generates the first generation
def initial_population():               
    population = []						
    while len(population) < 50:
        chromosome = generate_maze()
        p = mazeAstarSolver(chromosome)
        if(p.solve()[0]):               #check whether the generated maze is solvable. if yes then it's considered
            population.append(chromosome)
    return population					#returns 50 solvable mazes as the first generation

#function that starts the evolution process
def evolve(pop, c):
    while c < 30:						#30 generations are considered
        next_generation = []
        parent1, hs = selection(pop, 'n')    #hardest maze from current generation is chosen as the first parent
        generation.append(parent1)
        hardness_score.append(hs)
        remove_from_npa(pop, parent1)        
        parent2 = selection(pop, 'n')[0]	 #second hardest maze is chosen as the second parent
        for i in range(50):
			#the parents are crossed over to get the offspring and it's mutated
			#50 such offsprings are produced and they form the next generation
            next_generation.append(mutate(crossover(parent1, parent2)))        
        c += 1
        return evolve(next_generation, c)

def crossover(parent1, parent2):   #producing offsprings from parents
    length = len(parent1)
    point = random.randint(0, length - 1)     #a random point is chosen
	# offspring is formed by adding elemnts(chromosomes) of the first parent till the crossover point
	# and the rest is formed by adding elements(chromosomes) of the second parent
    child = np.append(parent1[:point], parent2[point:])    
    return child

def mutate(child):                 #mutation of offspring chromosomes
    for i in range(len(child)):
        rand_prob = round(random.uniform(0,1),3)   #a probability is chosen as random
        if rand_prob < 0.015:       #is mutation probability is less than 0.015
            child[i] = child[i] ^ 1 #then that chromosome is mutated. Here mutation means flipping 1 to 0 or 0 to 1
    return child
    
        
def display_evolution():           #display all generations along with the max hardness score of each generation
    for i in range(len(generation)):
        print('Generation: ',i+1, ' Max Hardness Score: ', hardness_score[i])

        
def fittest_of_all():              #displays in which generation the hardest maze when compared to other generations is found in 
    pos = -1
    hf = 0
    for i in range(len(hardness_score)):
        if hardness_score[i] > hf:
            hf = hardness_score[i]
            pos = i
    print('The hardest maze is found in Generation ',pos, ' with hardness score of ', hf)


generation = []
hardness_score = []

c = 0
evolve(initial_population(), c)


display_evolution()
fittest_of_all()



