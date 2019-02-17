import numpy as np
import queue

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

def reset_maze(maze):
    dim = maze.shape[0]
    maze[maze==CELL_PATH] = CELL_OPEN
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

    # Keep track of maximum size of fringe
    max_fringe_size = len(toVisit)

    # Moves counter
    num_moves = 0

    # Iterate through the fringe until it is empty
    while len(toVisit) > 0:
        coords = toVisit.pop()
        row = coords[0]
        col = coords[1]
        cell = maze[row][col]

        fringe_size = len(toVisit)
        if fringe_size > max_fringe_size:
            max_fringe_size = fringe_size

        num_moves = num_moves + 1

        # check if we reached the goal cell
        if row is dim-1 and col is dim-1:
            path.append(coords)
            break
        elif cell == CELL_BLOCKED or cell == CELL_PATH:
            continue
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
                    prev_coords = path.pop()
                path.append(prev_coords)
            
            # add to path
            path.append(coords)
            maze[row][col] = CELL_PATH

    # Check if last element in path is the goal node - if so we mark it as successful
    search_success = False
    if len(path) > 0:
        last = path[len(path)-1]
        if last[0] == dim-1 and last[1] == dim-1:
            search_success = True

    reset_maze(maze)

    result = {
        'status': search_success,
        'num_moves': num_moves,
        'max_fringe_size': max_fringe_size,
        'path': path
    }

    return result

# TreeNode for each cell in path; reverse order to find trace of path
class PathNode:
    def __init__(self, row, col, parent):
        self.row = row
        self.col = col
        self.parent = parent

# perform bfs search on a Maze to find a solution
def bfs_search(maze):
    dim = maze.shape[0]
    
    # Head of tree to keep track of all paths in BFS Search
    start = PathNode(0,0,None)

    # Declare fringe as queue for dfs search - start by looking down, and to the right
    toVisit = queue.Queue()
    toVisit.put(PathNode(1,0,start))
    toVisit.put(PathNode(0,1,start))

    # Keep track of maximum size of fringe
    max_fringe_size = toVisit.qsize()
    
    # Moves counter
    num_moves = 0

    # Store reference to last visited path node
    ptr = None

    # Iterate through fringe until it's empty
    while not toVisit.empty():
        ptr = toVisit.get()
        row = ptr.row
        col = ptr.col

        cell = maze[row][col]

        fringe_size = toVisit.qsize()
        if fringe_size > max_fringe_size:
            max_fringe_size = fringe_size
        
        num_moves = num_moves + 1

        if row == dim-1 and col == dim-1:
            break
        elif cell == CELL_BLOCKED or cell == CELL_PATH:
            continue
        else:
            # We've reached an open node that is not a goal node
            # Let's further explore this route and add it's children to the queue

            # Push one cell down to queue
            if row < dim-1:
                n_row = row+1
                n_col = col
                n_cell = maze[n_row][n_col]
                if n_cell != CELL_BLOCKED:
                    toVisit.put(PathNode(n_row,n_col,ptr))

            # Push one cell right to queue
            if col < dim-1:
                n_row = row
                n_col = col+1
                n_cell = maze[n_row][n_col]
                if n_cell != CELL_BLOCKED:
                    toVisit.put(PathNode(n_row,n_col,ptr))
            
            # Push one cell up to queue
            if row > 0:
                n_row = row-1
                n_col = col
                n_cell = maze[n_row][n_col]
                if n_cell != CELL_BLOCKED:
                    toVisit.put(PathNode(n_row,n_col,ptr))

            # Push one cell left to queue
            if col > 0:
                n_row = row
                n_col = col-1
                n_cell = maze[n_row][n_col]
                if n_cell != CELL_BLOCKED:
                    toVisit.put(PathNode(n_row,n_col,ptr))

            # Mark cell as visited
            maze[row][col] = CELL_PATH

    reset_maze(maze)

    # Iterate through tree and find path 
    path = []
    search_success = False
    if ptr is not None:
        # If last node's location is at the bottom right, we've reached the goal node
        if ptr.row == dim-1 and ptr.col == dim-1:
            search_success = True
            while ptr is not None:
                # Redraw path in maze grid
                path.insert(0, [ptr.row, ptr.col])
                ptr = ptr.parent

    result = {
        'status': search_success,
        'num_moves': num_moves,
        'max_fringe_size': max_fringe_size,
        'path': path
    }

    return result