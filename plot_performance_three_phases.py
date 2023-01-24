import re
import numpy as np
import pickle, os, time
from tqdm import tqdm
import torch
from simulator import Simulator
import matplotlib.pyplot as plt

import multiprocessing as mp
from parameter import *
from function import *



phase1 = pickle.load(open("saves/RESULTS_PHASE_1.pkl", "rb"))
phase2 = pickle.load(open("saves/RESULTS_PHASE_2.pkl", "rb"))
phase3 = pickle.load(open("saves/RESULTS_PHASE_3.pkl", "rb"))


x1 = np.array([np.array([res['org_utility'], res['sim_utility']]) for res in phase1])
x2 = np.array([np.array([res['org_utility'], res['sim_utility']]) for res in phase2])
x3 = np.array([np.array([res['org_utility'], res['sim_utility']]) for res in phase3])


plt.hist(x1[:,-1],bins=50,cumulative=True,density=True, histtype='step', label='original')
plt.hist(x2[:,-1],bins=50,cumulative=True,density=True, histtype='step', label='attacked')
plt.hist(x3[:,-1],bins=50,cumulative=True,density=True, histtype='step', label='robust')
plt.legend()
plt.savefig("results/performance_cdf_three_phases.pdf", format = 'pdf', dpi=300)
plt.show()

print('done')











