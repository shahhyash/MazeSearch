import numpy as np
import visualizer as viz

# Constants used for improved code readability
CELL_BLOCKED=0
CELL_OPEN=1
CELL_PATH=2

# Generate a Maze given a dimension (dim) and probability of cell being blocked (p)
def generate_maze(dim, p):
    maze = np.random.choice([CELL_BLOCKED, CELL_OPEN], (dim,dim), p=[p,1-p])
    maze[0][0] = CELL_PATH
    maze[dim-1][dim-1] = CELL_PATH
    return maze

def is_adjacent(coords1, coords2):
    row1 = coords1[0]
    col1 = coords1[1]

    row2 = coords2[0]
    col2 = coords2[1]

    if row1 == row2 and col1 == col2+1:
        return True
    elif row1 == row2 and col2 == col1+1:
        return True
    elif col1 == col2 and row1 == row2+1:
        return True
    elif col1 == col2 and row2 == row1+1:
        return True
    else:
        return False

# perform dfs search on a Maze to find a solution
def dfs_search(maze):
    dim = maze.shape[0]
    
    # Estabilish start point of stack - looking down, and to the right
    toVisit = [[0,1], [1,0]]
    
    # Array to keep track of path from start to end
    path = [[0,0]]

    # Moves counter
    num_moves = 0

    clear_visited = []

    # Iterate through the fringe until it is empty
    while len(toVisit) > 0:
        coords = toVisit.pop()
        row = coords[0]
        col = coords[1]
        cell = maze[row][col]

        num_moves = num_moves + 1

        # check if we reached the goal cell
        if row is dim-1 and col is dim-1:
            path.append(coords)
            break
        elif cell == CELL_BLOCKED:
            continue
        elif cell == CELL_PATH:
            if(len(toVisit) < 1):
                break

            # If we've already visited this node, and this node is adjacent to the next block to visit,
            # then, we've encountered a dead-end path and we can go ahead and clear those nodes from the path array

            next_coords = toVisit[len(toVisit)-1]
            if is_adjacent(coords, next_coords):
                while len(path) > 0:
                    prev_coords = path.pop()
                    prev_row = prev_coords[0]
                    prev_col = prev_coords[1]
                    if prev_row == row and prev_col == col:
                        path.append(prev_coords)
                        break
                    else:
                        clear_visited.append(prev_coords)
        else:
            # Here the cell is open so let's keep exploring further
            # we should add cells to this list so that we're exploring down/right, THEN left/up
            
            # Push one cell left to stack
            if col > 0:
                n_row = row
                n_col = col-1
                n_cell = maze[n_row][n_col]
                if n_cell != CELL_BLOCKED:
                    toVisit.append([n_row, n_col])
            
            # Push one cell up to stack
            if row > 0:
                n_row = row-1
                n_col = col
                n_cell = maze[n_row][n_col]
                if n_cell != CELL_BLOCKED:
                    toVisit.append([n_row, n_col])

            # Push one cell right to stack
            if col < dim-1:
                n_row = row
                n_col = col+1
                n_cell = maze[n_row][n_col]
                if n_cell != CELL_BLOCKED:
                    toVisit.append([n_row, n_col])
            
            # Push one cell down to stack
            if row < dim-1:
                n_row = row+1
                n_col = col
                n_cell = maze[n_row][n_col]
                if n_cell != CELL_BLOCKED:
                    toVisit.append([n_row, n_col])

            # Check if previous node in path is adjacent to one being added
            # if not, remove from path until you get to node that is adjacent to it 
            if len(path) > 0:
                prev_coords = path.pop()
                while not is_adjacent(prev_coords, coords) and len(path) > 0:
                    clear_visited.append(prev_coords)
                    prev_coords = path.pop()
                path.append(prev_coords)
            
            # add to path
            path.append(coords)
            maze[row][col] = CELL_PATH

    # For visited nodes that are not part of the path, let's remove them from the figure
    while len(clear_visited) > 0:
        clear_coords = clear_visited.pop()
        clear_row = clear_coords[0]
        clear_col = clear_coords[1]
        maze[clear_row][clear_col] = CELL_OPEN

    # Check if last element in path is the goal node - if so we mark it as successful
    search_success = False
    if len(path) > 0:
        last = path[len(path)-1]
        if last[0] == dim-1 and last[1] == dim-1:
            search_success = True

    result = {
        'status': search_success,
        'num_moves': num_moves,
        'path': path
    }

    return result
    
maze = generate_maze(10, 0.2)

viz.visualize(maze)

result = dfs_search(maze)
if result['status']:
    viz.visualize(maze)
else:
    print("Maze is not solvable.")
