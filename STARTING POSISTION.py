import numpy as np
import matplotlib.pyplot as plt
import pandas as panda

## this file has to be able to add a stochastic array over a given map,
## together with foodsources (FS)

Length = range(0,10,1)
Height = range(0,10,1)
x_list = []
y_list = []

# Generating a matrix of points with some noise added to them with a for loop,
# this still has to be able to be put over a map; will add later functionality.
# 
for u in Height:
    for i in Length:
        #make a point and add random noise to it np.random.normal()
        x_list.append(u + np.random.normal(0, 0.1))
        y_list.append(i + np.random.normal(0, 0.1))

# plot the points by extracting from the library
plt.scatter(x_list, y_list, marker='.', linewidths = 0.01, c='red')
# For every point [x,y] we will connect it to its right- and below-neighbour using plot
for o in x_list:
    for p in y_list:
        # define neighbours
        nb_center = [o, p]
        nb_right = [o, p+1]
        nb_below = [o+1, p]

        ## Drawing the lines between neighbours using plot. While keeping in mind the limits of the matrix
        # draw HORIZONTAL line
        if nb_center[1]<len(Length):
            plt.plot(nb_center, nb_right, mfc='red', markersize=0.1,)
        # draw VERTICAL line
        if nb_center[0]<len(Height):
            plt.plot(nb_center, nb_below, mfc='red', markersize=0.1,)
plt.show()