import numpy as np
from queue import PriorityQueue
import matplotlib.pylab as plt
import vizualiser as viz


class mazeCell(object):
    def __init__(self, x, y, open):
        
        self.open = open
        self.x = x
        self.y = y
        self.parent = None
        self.g = 0
        self.h = 0
        self.f = 0


class mazeSolver(object):
    def __init__(self):
        self.fringe = PriorityQueue()
        self.visited = set()
        self.cells = []
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
        
    def init_maze(self):
        self.a = self.generate_maze(10, 0.3)
        print(self.a)
        for x in range(self.dim):
            for y in range(self.dim):
                if self.a[x][y] == 0:
                    open = False
                else:
                    open = True
                self.cells.append(mazeCell(x, y, open))
        self.start = self.get_cell(0, 0)
        self.end = self.get_cell(self.dim - 1, self.dim - 1)
        
    def get_h(self, cell):
        return 10 * (abs(cell.x - self.end.x) + abs(cell.y - self.end.y))
    
    def get_cell(self, x, y):
        return self.cells[x * self.dim + y]
    
    def get_adjacent_cells(self, cell):
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
            
        viz.visualize(self.a)
            
    def update_cell(self, adj, cell):   
        adj.g = cell.g + 10
        adj.h = self.get_h(adj)
        adj.parent = cell
        adj.f = adj.h + adj.g
        
    def check_in_fringe(self, adj_cell):
        q = self.fringe.queue
        for i in range(len(q)):
            if q[i][0] == adj_cell.f and q[i][2].x == adj_cell.x and q[i][2].y == adj_cell.y:
                return True
            else:
                return False
        
        
    def solve(self):
        
        self.init_maze()
        self.fringe.put((self.start.f, 0, self.start))
        i = 1
        while not self.fringe.empty():
            f, x, cell = self.fringe.get()
            self.visited.add(cell)
            if cell is self.end:
                print ('self.end ',self.finish)
                self.finish = True
                return self.display_path()
            adj_cells = self.get_adjacent_cells(cell)
            for adj_cell in adj_cells:
                if adj_cell.open and adj_cell not in self.visited:
                    if self.check_in_fringe(adj_cell):
                        if adj_cell.g > cell.g + 10:
                            self.update_cell(adj_cell, cell)
                    else:
                        self.update_cell(adj_cell, cell)
                        self.fringe.put((adj_cell.f, i, adj_cell))
                      
                        i += 1                
        if(self.finish == False):
            print('No Solution!')
                    

def main():
	b=mazeSolver()
	b.solve()


