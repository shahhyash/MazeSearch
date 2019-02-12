import numpy as np
import visualizer as viz

CELL_BLOCKED=0
CELL_OPEN=1
CELL_PATH=2

# Generate a Maze given a dimension (dim) and probability of cell being blocked (p)
def generate_maze(dim, p):
    maze = np.random.choice([CELL_BLOCKED, CELL_OPEN], (dim,dim), p=[p,1-p])
    maze[0][0] = CELL_PATH
    maze[9][9] = CELL_PATH
    return maze

# perform dfs search on a Maze to find a solution
def dfs_search(maze):
    dim = maze.shape[0]
    
    # Estabilish start point of stack - looking down, and to the right
    toVisit = [[0,1], [1,0]]

    # Iterate through the fringe until it is empty
    while len(toVisit) > 0:
        coords = toVisit.pop()
        row = coords[0]
        col = coords[1]
        cell = maze[row, col]

        print("Row %d, Col %d, Value %d" % (row, col, cell))

        # check if we reached the goal cell
        if row is dim-1 and col is dim-1:
            break
        elif cell == CELL_BLOCKED or cell == CELL_PATH:
            # either cell is blocked or it's reached a point which we've already explored as the path
            print("reached cell that is either blocked or hit the same path")
        else:
            # Here the cell is open so let's keep exploring further
            # we should add cells to this list so that we're exploring down/right, THEN left/up
            
            # Push one cell up to stack
            if row > 0:
                toVisit.append([row-1, col])
            
            # Push one cell left to stack
            if col > 0:
                toVisit.append([row, col-1])

            # Push one cell right to stack
            if col < dim-1:
                toVisit.append([row, col+1])
            
            # Push one cell down to stack
            if row < dim-1:
                toVisit.append([row+1, col])

            maze[row][col] = CELL_PATH

maze = generate_maze(10, 0.2)

viz.visualize(maze)

dfs_search(maze)

viz.visualize(maze)
