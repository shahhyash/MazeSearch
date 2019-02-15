import visualizer as viz
import numpy as np
from queue import PriorityQueue
import matplotlib.pylab as plt
import math

class mazeCell(object):
    def __init__(self, x, y, open):        
        self.open = open      #True if accessible, False if blocked
        self.x = x            #x co-ordinate of the cell
        self.y = y            #y co-ordinate of the cell
        self.parent = None
        self.h = 0
        self.f = 0

class mazeSolver(object):
    def __init__(self):
        self.fringe = PriorityQueue()    
        self.visited = set()   #keeps track of visited cells
        self.cost = {}         #dict containing all the cells considered along with updated cost
        self.cells = []        #list of all cells
        self.dim = 0
        self.finish = False
        
    def generate_maze(self, dim, p):
        CELL_BLOCKED=0
        CELL_OPEN=1
        CELL_PATH=2
        self.dim = dim
        maze = np.random.choice([CELL_BLOCKED, CELL_OPEN], (dim,dim), p=[p,1-p])
        maze[0][0] = CELL_PATH
        maze[dim-1][dim-1] = CELL_PATH
        return maze
        
    def init_maze(self, dim, p):        
        self.a = self.generate_maze(dim, p)
        for x in range(self.dim):
            for y in range(self.dim):
                if self.a[x][y] == 0:
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
          
    def display_path(self):
        cell = self.end        
        while cell.parent is not self.start:
            cell = cell.parent
            self.a[cell.x][cell.y] = 2            
            print ('path: cell:',cell.x, ',', cell.y)            
        visualize(self.a)
                   
    def solve(self):  
        dim = int(input('Enter the dimension of the maze: '))
        p = float(input('Enter the probability at which the blocks will be generated: '))
        self.init_maze(dim, p)
        heu = input('Enter type of heuristic (\'m\' for manhattan/ \'e\' for euclidean): ')
        self.fringe.put((self.start.f, 0, self.start))  #adding start cell to the fringe
        self.cost[self.start] = 0
        i = 1
        while not self.fringe.empty():
            f, x, cell = self.fringe.get()
            self.visited.add(cell)       #add cells to visited set as and when they're visited
            if cell is self.end:         #if popped cell is target cell, then path has been found
                self.finish = True
                return self.display_path()
                #return self.finish
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
                        i += 1
             
        if(self.finish == False):
            viz.visualize(self.a)
            print('No Solution!')
            
        #return self.finish


#for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
#    total = 0
#    success_count = 0
#    for i in range(5):
#        sol = mazeSolver()
#        solution_found = sol.solve(p)
#        print(solution_found)
#        if solution_found:
#            success_count += 1
#        total += 1
#    print('For p = ',p,' total simulations = ', total, ' ,success = ', success_count)


sol = mazeSolver()
sol.solve()




