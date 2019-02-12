import matplotlib.pylab as plt
import numpy as np

def visualize(data):
    plt.style.use('ggplot')
    plt.rcParams["axes.axisbelow"] = False

    fig = plt.figure()
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

    plt.axes().set_aspect('equal') #set the x and y axes to the same scale
    plt.axes().invert_yaxis() #invert the y-axis so the first row of data is at the top3

    plt.show()

def test_visualizer():
    maze = np.ones((10,10))
    maze[0][2] = 0
    maze[4][2] = 0

    maze[0][0] = 2
    maze[1][0] = 2
    maze[2][0] = 2
    maze[3][0] = 2
    maze[4][0] = 2
    maze[5][0] = 2
    maze[6][0] = 2
    maze[7][0] = 2
    maze[7][1] = 2
    maze[7][2] = 2

    maze[8][2] = 2
    maze[9][2] = 2
    maze[9][3] = 2
    maze[9][4] = 2
    maze[9][5] = 2
    maze[9][6] = 2
    maze[9][7] = 2
    maze[9][8] = 2
    maze[9][9] = 2

    visualize(maze)