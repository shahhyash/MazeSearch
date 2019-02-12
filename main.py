import numpy as np
import visualizer as viz

def generate_maze(dim, p):
    maze = np.random.choice([0,1], (dim,dim), p=[p,1-p])
    maze[0][0] = 2
    maze[9][9] = 2
    return maze

maze = generate_maze(10, 0.2)

viz.visualize(maze)