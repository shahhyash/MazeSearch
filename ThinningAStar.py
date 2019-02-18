
import numpy as np
from queue import PriorityQueue
import matplotlib.pylab as plt
import math

def visualize(data):
    plt.style.use('ggplot')
    plt.rcParams["axes.axisbelow"] = False

    fig = plt.figure(figsize = (8,8))
    
    ax = fig.add_subplot(111)

    ax.pcolormesh(data, cmap='RdGy_r', zorder=1) # RdGy allows us to define gray blocks and red path blocks
    ax.grid(True, color="black", lw=1)           # Grid lines to identify squares in maze

    # set range of ticks to show entire grid
    ticks = np.arange(0, data.shape[0], 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # remove ticks
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.set_aspect('equal') #set the x and y axes to the same scale
    ax.invert_yaxis() #invert the y-axis so the first row of data is at the top3

    plt.show()


# In[3]:


class mazeCell(object):
    def __init__(self, x, y, open):        
        self.open = open      #True if accessible, False if blocked
        self.x = x            #x co-ordinate of the cell
        self.y = y            #y co-ordinate of the cell
        self.parent = None
        self.h = 0
        self.f = 0
     


# In[29]:


class mazeSolver(object):
    def __init__(self, maze, heu):
        self.fringe = PriorityQueue()    
        self.visited = set()   #keeps track of visited cells
        self.cost = {}         #dict containing all the cells considered along with updated cost
        self.cells = []        #list of all cells
        self.dim = 10
        self.finish = False
        self.length = 0
        self.max_frig_size = 0
        self.max_nodes = 0
        self.a = maze
        self.qmaze = maze
        self.heu = heu        
        
    def init_maze(self):        
        #self.a = self.generate_maze(dim, p)
        #print(self.dim)
        self.dim = self.a.shape[0]
        print(self.a)
        
        for x in range(self.dim):
            for y in range(self.dim):
                if self.a[x][y] == 2:
                    self.a[x][y] = 1
                if self.a[x][y] == 0:
                    open = False
                else:
                    open = True
                self.cells.append(mazeCell(x, y, open)) #adding all cells to the 'cells' list
        self.a[0][0] = 2
        self.a[self.dim - 1][self.dim - 1] = 2
        #visualize(self.a)
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
        self.length = 0
        while cell.parent is not self.start:
            self.length += 1
            cell = cell.parent
            self.a[cell.x][cell.y] = 2            
            print ('path: cell:',cell.x, ',', cell.y)            
        
        self.length += 2
        print('Length of shortest path: ',self.length)
        print('Maximal fringe size: ',self.max_frig_size)
        print('No. of nodes expanded: ',self.max_nodes)
        return(self.length)
    
    def shortest_path_length(self):
        cell = self.end 
        self.length = 0
        while cell.parent is not self.start:
            self.length += 1
            cell = cell.parent           
            #print ('path: cell:',cell.x, ',', cell.y)            
        
        self.length += 2
        return(self.length)
        
    def solve(self):        
        self.init_maze()
        #remove fraction qmaze
        self.qmaze = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 0, 1, 0],
 [1, 1, 1, 1, 0, 1, 1, 0],
 [1, 0, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 1, 0, 1, 1, 0],
 [1, 1, 1, 1, 1, 1, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1]])
        #self.start = self.get_cell(3,0)
        #heu = input('Enter type of heuristic (\'m\' for manhattan/ \'e\' for euclidean): ')
        self.fringe.put((self.start.f, 0, self.start))  #adding start cell to the fringe
        self.max_frig_size = self.fringe.qsize()
        #print(self.fringe.qsize())
        
        self.cost[self.start] = 0
        i = 1
        while not self.fringe.empty():
            f, x, cell = self.fringe.get()
            self.visited.add(cell)       #add cells to visited set as and when they're visited
            if cell is self.end:         #if popped cell is target cell, then path has been found
                self.finish = True
                self.max_nodes = i
                lk = self.display_path()
                visualize(self.a)
                return self.finish
            adj_cells = self.get_adjacent_cells(cell)
            for adj_cell in adj_cells:
                if adj_cell.open and adj_cell not in self.visited:
                    c = self.cost[cell]   
                    if adj_cell not in self.cost or self.cost[adj_cell] > c:
                    #if cell hasn't been seen before or has worse cost than the new cost
                        self.cost[adj_cell] = c;          #update to new cost
                        adj_cell.parent = cell         
                        f = astar_heuristic(adj_cell.x, adj_cell.y, self.qmaze) #get length of path from the adj_cell in the modified maze
                        self.fringe.put((f,i,adj_cell))   #push in fringe with f as priority
                        if(self.fringe.qsize() > self.max_frig_size):
                            self.max_frig_size = self.fringe.qsize()
                        i += 1
            #print(self.fringe.qsize())
      
             
        if(self.finish == False):
            print('No Solution!')
            
        self.max_nodes = i
        
    def mod_astar(self, x, y):        
        self.init_maze()
        self.start = self.get_cell(x,y)
        #heu = input('Enter type of heuristic (\'m\' for manhattan/ \'e\' for euclidean): ')
        self.fringe.put((self.start.f, 0, self.start))  #adding start cell to the fringe
        self.max_frig_size = self.fringe.qsize()
        #print(self.fringe.qsize())
        
        self.cost[self.start] = 0
        i = 1
        while not self.fringe.empty():
            f, x, cell = self.fringe.get()
            self.visited.add(cell)       #add cells to visited set as and when they're visited
            if cell is self.end:         #if popped cell is target cell, then path has been found
                self.finish = True
                self.max_nodes = i
                lk = self.shortest_path_length()
                return lk
            adj_cells = self.get_adjacent_cells(cell)
            for adj_cell in adj_cells:
                if adj_cell.open and adj_cell not in self.visited:
                    c = self.cost[cell] + 10   #adding g score
                    if adj_cell not in self.cost or self.cost[adj_cell] > c:
                    #if cell hasn't been seen before or has worse cost than the new cost
                        self.cost[adj_cell] = c;          #update to new cost
                        adj_cell.parent = cell         
                        f = c + self.get_h(adj_cell, self.heu) #get f value
                        self.fringe.put((f,i,adj_cell))   #push in fringe with f as priority
                        if(self.fringe.qsize() > self.max_frig_size):
                            self.max_frig_size = self.fringe.qsize()
                        i += 1
            #print(self.fringe.qsize())
      
             
        if(self.finish == False):
            print('No Solution!')
            
        self.max_nodes = i
        
    
       

def astar_heuristic(x, y, qmaze):
    temp = mazeSolver(qmaze, 'm')
    dim = qmaze.shape[0]
    zero = int(0)
    print('Calling ',x,',',y)
    if x == dim - 1 and y == dim - 1:
        return zero
    else:
        return(temp.mod_astar(x,y))
    


hjk = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
 [0, 1, 1, 1, 1, 0, 1, 0],
 [0, 1, 1, 1, 0, 1, 1, 0],
 [1, 0, 1, 1, 1, 1, 1, 1],
 [1, 0, 1, 1, 0, 1, 1, 0],
 [1, 1, 1, 1, 1, 0, 0, 1],
 [1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1]])



jk = mazeSolver(hjk, 'm')


jk.solve()



