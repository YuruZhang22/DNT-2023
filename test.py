import torch
from bayesian_torch.dnn import DNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ue_speed = np.array([1, 2, 3])
scope = np.array([3, 6, 10])     #1,2,3 represent the radius of UE activity in 3m, 6m, 10m
column = 7

all_pmf_flexran_mcs = []
def read_mcs_from_flexran(ue_speed, scope):
    # data = np.loadtxt("/home/ins/Downloads/MCS_from_NS3/DlMacStats_speed"+str(ue_speed)+"_UE"+str(scope)+".txt", skiprows = 1, dtype = np.float32)
    data = np.loadtxt("/home/ins/Downloads/MCS_from_flexran/scale_"+str(scope)+"_speed_"+str(ue_speed)+".txt", delimiter =',', dtype = np.int0)
    return data 

def calculate_pmf(mcs):
    distribution,_ = np.histogram(mcs, bins = 15, range = (0,29))
    pmf = distribution / len(mcs)
    return pmf

for i in ue_speed:
    for j in scope:

        flexran_mcs = read_mcs_from_flexran(i, j)
        pmf_flexran_mcs = calculate_pmf(flexran_mcs)
        all_pmf_flexran_mcs.append(pmf_flexran_mcs)
        # print(pmf_flexran_mcs)