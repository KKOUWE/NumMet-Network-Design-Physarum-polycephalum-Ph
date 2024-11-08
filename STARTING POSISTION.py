import numpy as np
import matplotlib.pyplot as plt
import pandas as panda
import networkx as nx

## this file has to be able to add a stochastic array of points over a given map,
## together with foodsources (FS)

# Create an empty graph
G = nx.Graph()

# Define the grid dimensions
rows, cols = 10, 10
min_distance = np.sqrt(2)
# Add nodes in a 10x10 grid
for i in range(rows):
    for j in range(cols):
        G.add_node((i, j))
        # Set up positions for a 10x10 grid
        # standerd deviation
        sd = 0.2
        pos = {(i, j): (j + np.random.normal(loc=0,scale=sd, size=None) , -i + np.random.normal(loc=0,scale=sd, size=None) ) for i in range(rows) for j in range(cols)}
        # Check all other nodes in the grid to see if they meet the distance condition # lots of room for optimizatin in this loop
        for x in range(rows):
            for y in range(cols):
                pos_center = pos.get((i,j))
                pos_nb = pos.get((x,y))
                if pos_nb != pos_center:
                    pos_center_i = pos_center[0]
                    pos_center_j = pos_center[1]
                    pos_nb_i = pos_nb[0]
                    pos_nb_j = pos_nb[1]
                # Calculate distance
                    distance = np.sqrt((pos_center_i- pos_nb_i) ** 2 + (pos_center_j - pos_nb_j) ** 2)
                    if distance <= min_distance:
                        G.add_edge((i, j), (x, y))



# Draw the graph
plt.figure()
nx.draw(G, pos, node_size=5, node_color="red", with_labels=False)
plt.show()
## It works !!

## Now we need to implement food sources = FS and the tube size = TS. Also this tube size has to 
# change with every iteration dt. according to 