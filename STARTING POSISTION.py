import numpy as np
import matplotlib.pyplot as plt
import pandas as panda
import networkx as nx
import plotly.graph_objects as go
import random
from itertools import islice

## this file has to be able to add a stochastic array of points over a given map,
## together with foodsources (FS)

# Create an empty graph
G = nx.Graph()

# Define the grid dimensions and constants
rows, cols = 5,5
min_distance = np.sqrt(2)       # bc network isnt rly scaled, Random interconnection around sqrt 2 gives good anwsers 
r = 1                               # starting radius for all tubes
mu = 1                              # dynamic viscocity of water (not 1 but yk)
pos_dict = {}                            # posistion dictionary for every node
sd = 0.15                            # standard deviation
# Add nodes in a 10x10 grid
for i in range(rows):
    for j in range(cols):
        # Set up positions for a 10x10 grid
        # standerd deviation
        pos_i = j + np.random.normal(loc=0,scale=sd, size=None) 
        pos_j = -i + np.random.normal(loc=0,scale=sd, size=None) 
        pos_dict[(i, j)] = (pos_i , pos_j )
        G.add_node((i, j), pos=(pos_i, pos_j), FS = 0)
# Retrieve the positions for all nodes
pos = nx.get_node_attributes(G, 'pos')

# Check all other nodes in the grid to see if they meet the distance condition 
# lots of room for optimizatin in this loop
for i in range(rows):
    for j in range(cols):
        for x in range(rows):
            for y in range(cols):
                pos_center = pos_dict.get((i,j))
                pos_nb = pos_dict.get((x,y))
                if pos_nb != pos_center:
                    pos_center_i = pos_center[0]
                    pos_center_j = pos_center[1]
                    pos_nb_i = pos_nb[0]
                    pos_nb_j = pos_nb[1]
                # Calculate distance
                    distance = np.sqrt((pos_center_i- pos_nb_i) ** 2 + (pos_center_j - pos_nb_j) ** 2)
                    if distance < min_distance:
                        G.add_edge((i, j), (x, y), length = distance, radius = r)




# Draw the graph
plt.figure()
nx.draw(G, pos, node_size=5, node_color="red", with_labels=True)
plt.show()
# ## It works !!

## Now we need to implement food sources = FS and the tube size = TS. Also this tube size has to 
# change with every iteration dt. according to the paper.

## FOOD SOURCES (FS)
# lets try one FS to start with.
#  make a second dict but now with FS values; set all to zero and manually overwrite one

# for i in range(rows):
#     for j in range(cols):
#         FS = {(i,j) : (0,0) for i in range(rows) for j in range(cols)}
FS = {}
FS[(0,0)] = 10
FS[(0,4)] = 10
FS[(4,0)] = 10
FS[(4,4)] = 10
FS[(2,2)] = 10

print(f"The nodes containing food are: {FS}")

## ITERATION OF FLOW: we need a few paramaters defined;

# Q_ij = pi*r^4(p_j-p_i)/8*mu*L_ij      -Poisseuille flow
# here D_ij(r) could be pi*r^4/8*mu      - this would be the variable responding to the usage of tube.
# L_ij is the edge lenght of two nodes i and j.
# p_i is set as the only source of positive pressure and p_j is ?zero? 

# Every iteration we:
# 1.Select source and sink nodes
# 2.Find possible paths
# 3.Find Q_ij
# 4.adjsut r's
# 5.repeat

# We are going to limit the different paths by assumimg the mold has a sense of direction and by adhereing to the rule:
# the sum of the outflows at Qsource is equal to the sum of the inflows at Qsink. Meaning if the source has 5 edges
# we will look for 5 different paths to the sink node and share flow over these paths accordingly.
# While also only considering a flow if the next node is closer to the sink node then preciously. 

# ---
#  1  Select Nodes
# ---
def select_source_and_sink_nodes(FS):
    FS_source = random.choice(list(FS.keys()))
    FS_sink = random.choice(list(FS.keys()))
    while FS_source == FS_sink:
        FS_source = random.choice(list(FS.keys()))
        FS_sink = random.choice(list(FS.keys()))
    Nodes = [FS_source, FS_sink]
    return Nodes


# ---
#  2  Find paths
# ---
# this is a build-in library of networkx
def k_shortest_paths(G, Nodes, k, weight='length'):
    paths_list = list(islice(nx.shortest_simple_paths(G, Nodes[0], Nodes[1], weight=weight), k))
    return  paths_list


# ---
#  3
# ---


# ---
#  4  Find path resistivity
# ---

# we need to make a function that returns the resitivity of a path.
# using the poisseuille definition of tube flow resistance: R = (8*L*mu)/(pi*r^4)
def Calc_Resistance(L,r):
    resistance = (8*L*mu)/(np.pi*(r)**4)
    return resistance

def Resistance_of_paths(paths_list):
    resistance_of_paths_list = [sum(Calc_Resistance(G[u][v]['length'], G[u][v]['radius']) for u, v in zip(path, path[1:]))
    for path in paths_list]
    return resistance_of_paths_list

# ---
#  4  Find flow distr.
# ---

# to find the distribuiton we not only need the total path taken but also the radius and lenght of every individual
# tube. This way we ca make an electrical circuit analogy of current distribution based on total resistance of a path.

def find_flow_distribution(resistance_of_paths_list):
    Q = {}
    for i in range(0,k,1):
        Q[i] = []
    return


# ---
#  T  Test code
# ---

# 1. Select nodes
Nodes = select_source_and_sink_nodes(FS)
print(f"the source node is: {Nodes[0]} and the sink node is: {Nodes[1]}")
# 2. Find paths
k = 4
list_of_paths = k_shortest_paths(G,Nodes,k)
print(f"the {k} shortest possible paths are:{list_of_paths}")
# 3. Calc resistance
resistance_of_path_list = Resistance_of_paths(list_of_paths)
print(f"their respective resistances are: {resistance_of_path_list}")
# 4. divide flow
plt.show()