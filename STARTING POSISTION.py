import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import random
from itertools import islice
import pickle
import os
import shutil

## this file has to be able to add a stochastic array of points over a given map,
## together with foodsources (FS)

# Create an empty graph
G = nx.Graph()

# Define the grid dimensions and constants
rows, cols = 50,50
min_distance = np.sqrt(2)+0.1       # bc network isnt rly scaled, Random interconnection around sqrt 2 gives good anwsers 
r = 1E-3                            # starting radius for all tubes
mu = 8.9E-4                         # dynamic viscocity of water (not 1 but yk)
pos_dict = {}                       # posistion dictionary for every node
sd = 0.2                            # standard deviation
I_0 = 6                           # Normalized flow/current
k = 6                               # number of paths
gamma = 1.8                        # constant that determines non linearity of radius response to flow. Present in dRdt function.         
t = 0                               # timestep initialisation
convergence = False                 # Convergence init.
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

# # Draw the graph
# plt.figure()
# nx.draw(G, pos, node_size=5, node_color="red", with_labels=True)
# plt.title(f"Network at t={t}")
# plt.show()

## Now we need to implement food sources = FS and the tube size = TS. Also this tube size has to 
# change with every iteration dt. according to the paper.

## FOOD SOURCES (FS)
FS = {}
FS[(0,0)] = 10
FS[(0,(cols-1))] = 10
FS[(round(rows/2),round(cols/5))] = 10
FS[(round(rows/2),round(cols*4/5))] = 10
FS[((rows-1),0)] = 10
FS[((rows-1),(cols-1))] = 10
#FS[(round(rows/2),round(cols/2))] = 10
# important parameter in tube reduction func

amount_FS = len(FS)   
print(amount_FS)
print(f"The {amount_FS} nodes containing food are: {FS}")
dR = (np.pi*r**4)/(8*mu)                 # reduction of radius term. present in dRdt function           
print(f'the adjuster dR term is:{dR}')

# extra list making for drawing purposes
FS_list = list(FS.keys())
## ITERATION OF FLOW: we need a few paramaters defined;

# Q_ij = pi*r^4(p_j-p_i)/8*mu*L_ij      -Poisseuille flow
# here D_ij(r) could be pi*r^4/8*mu      - this would be the variable responding to the usage of tube.
# L_ij is the edge lenght of two nodes i and j.
# p_i is set as the only source of positive pressure and p_j is ?zero? 

# Every iteration we:
# 1.Select source and sink nodes
# 2.Find possible paths
# 3.Find resistance
# 4.Find flow per tube
# 5.adjust r per tube
# 6.repeat


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
# we want to find the path of least resistance, so we have to add resistance to the edges as an attribute.
# Using the poisseuille definition of tube flow resistance: R = (8*L*mu)/(pi*r^4)

def Calc_Resistance_of_all_tubes():
    for u,v in G.edges():
        G[u][v]['resistance'] = (8*(G[u][v]['length'])*mu)/(np.pi*(G[u][v]['radius'])**4)
    return ## returns nothing, updates resistances of tubes

# this is a build-in library of networkx
def k_shortest_paths(G, Nodes, k, weight='resistance'):
    paths_list = list(islice(nx.shortest_simple_paths(G, Nodes[0], Nodes[1], weight=weight), k))
    return  paths_list


# ---
#  3  Find path resistivity
# ---
# we need to make a function that returns the resitivity of a path.
# The path is considered as a series of resistors, therefore we sum them.

def Resistance_of_paths(paths_list):        
    resistance_of_paths_list = [sum((G[u][v]['resistance']) for u, v in zip(path, path[1:]))
    for path in paths_list]
    return resistance_of_paths_list

# ---
#  4  Find flow per path and thus tube
# ---
# we can make an electrical circuit analogy of current distribution based on total resistance of a path.

def find_flow_distribution(resistance_of_paths_list):
    inverted_res_list = [(1/s) for s in resistance_of_paths_list]
    Q = {}
    for i in range(0,k,1):
        # calculate while consdidering parallel posistioning of tubes so: 1/Rt = 1/R1 + 1/R2 +...
        total_resistance_i = (sum(inverted_res_list) - inverted_res_list[i])**-1
        Q[i] = (total_resistance_i/(resistance_of_paths_list[i]+total_resistance_i))*I_0
    return Q

# now that we know the flow per path we need to keep track of the flow per tube:
def find_flow_per_tube(list_of_paths, Q):
    # initialise all edges with flow zero
    flow_per_tube = {tuple(sorted(edge)) : 0 for edge in G.edges()}
    a=0
    for path in list_of_paths:
        for u, v in zip(path, path[1:]):
            flow_per_tube[tuple(sorted((u,v)))] += Q.get(a)
        a+=1
    return flow_per_tube


# ---
#  5  Adjust r's
# ---
# We need to adjust the value of every r of every edge after every timestep iter to accomodate the 
# addapting behaviour of the mold

def Adjust_radius(flow_per_tube):
    # applies drdt function,sigmoid minus constant, to every edge.
    # the increase is reduced because it lowers the dependence on randomness of the first few calculations.
    for u,v in G.edges():
        G[u][v]['radius'] += 0.01*r*(((flow_per_tube.get((u,v)))**gamma)/(1+(flow_per_tube.get((u,v)))**gamma)) - dR*30     #scaled to r
        if G[u][v]['radius'] < 0:
            G[u][v]['radius'] = 0       # ensure radius min caps at 0 
    return # returns nothing, simply adjusts existing edge attributes 'radius'


# ---
#  6  Repeat timesteps iterations until network reaches certain convergence conditions
# ---
# we want to repeat all the above per timestep as long as some convergence condictions are not met.
# This means we have to define the convergence conditions.
# Create timestep functionallity.
# and add color functionality of plots with each iter, where we want to correlate edge thickness and opacity to the radius of a tube.

# Convergence conditions: if statement: if there is a path in k paths of least resistance that takes more than 0.99
# of the flow the network is converged.
def Check_convergence(convergence, Q): #both could work simultaniously
    # Single path convergence
    for i in range(k):
        if Q[i]>=0.99*I_0:
            convergence = True
            print("last figure!")
    #Time step convergence
    if t >= 7000:
        convergence = True
    return convergence

# Clearing the network storage folder:
directory_path = 'NumMet-Network-Design-Physarum-polycephalum-Ph/Network_iterations/pickle_testing'
if os.path.exists(directory_path):
    shutil.rmtree(directory_path)  # Remove the folder and all its contents
os.makedirs(directory_path)  # Recreate the folder


# ITER LOOP:
while convergence == False:
    print(f't={t}')
    # 1. Select nodes
    Nodes = select_source_and_sink_nodes(FS)
    #print(f"the source node is: {Nodes[0]} and the sink node is: {Nodes[1]}")

    # 2. Re-Calc resistance and Find paths
    next_res = Calc_Resistance_of_all_tubes()
    list_of_paths = k_shortest_paths(G,Nodes,k)
    #print(f"the {k} paths of least resistance are:{list_of_paths}")

    # 3. Calc resistance
    resistance_of_paths_list = Resistance_of_paths(list_of_paths)
    #print(f"their respective resistances are: {resistance_of_paths_list}")

    # 4. divide flow
    Q = find_flow_distribution(resistance_of_paths_list)
    #print(f'the distribution of flow trough each path is: {Q}')
    #print(f'Sum is one check: {sum(Q.values())}')
    flow_per_tube = find_flow_per_tube(list_of_paths, Q)
    #print(f'the amount of flow through each tube this iteration is: {flow_per_tube}')

    # 5. Adjust radius
    next_rad = Adjust_radius(flow_per_tube)
    radius_list = [G[u][v]['radius'] for u,v in G.edges]
    #print(f'the new radia are: {radius_list}')

    # Now we only really draw and save a plot every 100 timesteps
    if t%50 == 0:
        # Draw the graph
        plt.figure()
        plt.title(f"Network at t={t}")
        # normalize opacity
        max_r = max(radius_list)
        edge_opacity = [r/max_r for r in radius_list]       # normalized [0,1]
        edge_width = [(r/max_r) for r in radius_list]
        nx.draw(
        G,
        pos,
        with_labels = False,
        node_color = ['blue' if node in FS_list else'red' for node in G.nodes()] ,
        node_size = [50 if node in FS_list else 1 for node in G.nodes()] ,
        width = edge_width,
        )
        # storing of every network
        os.makedirs(directory_path, exist_ok= True)
        filename = os.path.join(directory_path,f"network_at_{t}.png")
        plt.savefig(filename)
        plt.close()

    # Convergence check
    convergence = Check_convergence(convergence,Q)
    t+=1

# ---
#  7 Loading and displaying
# ---
# Now that the networks are getting elaborate we want a way to run the network uniteruptedly, storing every iter.
# and displaying certain timesteps at the end.



# ---
#  8  Data collection from last network
# ---
# in order to be able to make some quantative statements we need two parameters of the network to be extracted.
# the cost parameter which will be represented by  the total length of the network. (TL)
# the avarage mean distance of the network. (MD)
# this will all be compared to the minimally spanning tree (MST), which we will compute first

# MST = nx.Graph()
# for node in FS_list:
#     MST.add_node(node,**G.nodes[node])

# pos_mst = nx.get_node_attributes(MST, 'pos')
# plt.figure()
# nx.draw(MST, pos_mst)
# plt.show()



# plt.figure(1)
# plt.hist(radius_list,bins=500)
# plt.title('histogram of radius data to chose a lower limit')
# plt.xlabel('radia in meter')
# plt.ylabel('occurence rate')
# plt.show()
# # pick LL
# lower_limit = 2*r

# # TL
# def Get_TL():
    
#     included_edges_list = []
#     for u,v in G.edges:
#         if G[u][v]['radius'] > lower_limit:
#             included_edges_list.append(G[u][v]['length'])
    
#     TL = sum(included_edges_list)
#     return TL
# TL = Get_TL()
# print(f'the total lentgh of the network is: {TL}')


# # to find MD we will first clean our final network by removing all edges that are under the lower limit, plus thus will
# # be a good visual check of our final network.
# for u,v in G.edges():
#     if G[u][v]['radius'] < lower_limit:
#         G.remove_edge(u,v)

# # Final plot
# plt.figure()
# plt.title(f"Network at t={t}")
#     # normalize opacity
# max_r = max(radius_list)
# edge_opacity = [r/max_r for r in radius_list]       # normalized [0,1]
# edge_width = [(r/max_r) for r in radius_list]
# nx.draw(
# G,
# pos,
# with_labels = False,
# node_color = ['blue' if node in FS_list else'red' for node in G.nodes()] ,
# node_size = [50 if node in FS_list else 1 for node in G.nodes()] ,
# width = edge_width,
# )
#     # storing network
# os.makedirs(directory_path, exist_ok= True)
# filename = os.path.join(directory_path,f"network_at_{t}.png")
# plt.savefig(filename)
# plt.show()

# ## get MD
# def Get_AMD():
#     distances_all_nodes = []
#     n = 0
#     for a in FS_list:
#         for b in FS_list:
#             if a !=b:
#                 md_length = nx.shortest_path_length(G, a, b, weight='length')
#                 distances_all_nodes.append(md_length)
#                 n += 1
#     AMD = sum(distances_all_nodes)/n
#     return AMD

# AMD = Get_AMD()
# print(f'the average minimal distance of the network is: {AMD}')

