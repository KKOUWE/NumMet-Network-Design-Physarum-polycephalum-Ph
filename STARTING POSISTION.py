import numpy as np
import matplotlib as plt
import pandas as panda

## this file has to be able to add a stochastic array over a given map,
## together with foodsources (FS)

Length = range(0,10000,1)
Height = range(0,10000,1)
points_dic = {}
for u in Height:
    for i in Length:
        #make a point and add random noise to it np.random.normal()