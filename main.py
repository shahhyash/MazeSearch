import numpy as np
import queue
import visualizer as viz
import SearchUtils

# Generate Maze
test_dim = 50
maze = SearchUtils.generate_maze(test_dim, 0.2)

# Show Maze before solving
viz.visualize(maze)

# Perform DFS Search
print("-----------------------------\nDFS Search:\n-----------------------------")
result = SearchUtils.dfs_search(maze)
if result['status']:
    print("num moves: %d" % result['num_moves'])
    print("maximum fringe size: %d" % result['max_fringe_size'])
    print("path length: ", len(result['path']))
    print("path: ", result['path'])
    viz.visualize(maze, result['path'])
else:
    print("num moves: %d" % result['num_moves'])
    print("Maze is not solvable.")

# Perform BFS Search
print("\n\n-----------------------------\nBFS Search:\n-----------------------------")
result = SearchUtils.bfs_search(maze)
if result['status']:
    print("num moves: %d" % result['num_moves'])
    print("maximum fringe size: %d" % result['max_fringe_size'])
    print("path length: ", len(result['path']))
    print("path: ", result['path'])
    viz.visualize(maze, result['path'])
else:
    print("num moves: %d" % result['num_moves'])
    print("Maze is not solvable.")

print("\n")

