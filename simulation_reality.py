# from bayesian_torch.bnn import BNN
# from bayes_opt import BayesianOptimization
import torch
from bayesian_torch.dnn import DNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ue_speed = np.array([1, 2, 3])
scope = np.array([3, 6, 10])     #scope represent the radius of UE activity in 3m, 6m, 10m
column = 7

def read_mcs_from_simulator(ue_speed, scope, column):
    # data = np.loadtxt("/home/ins/Downloads/MCS_from_NS3/DlMacStats_speed"+str(ue_speed)+"_UE"+str(scope)+".txt", skiprows = 1, dtype = np.float32)
    data = np.loadtxt("/home/ran/Downloads/MCS_from_NS3/scale_"+str(scope)+"_speed_"+str(ue_speed)+".txt", skiprows = 1, dtype = np.float32)
    return data[:,column-1]  

def read_mcs_from_flexran(ue_speed, scope):
    data = np.loadtxt("/home/ran/Downloads/MCS_from_flexran/scale_"+str(scope)+"_speed_"+str(ue_speed)+".txt", delimiter =',', dtype = np.int0)
    return data 

def calculate_pmf(mcs):
    distribution,_ = np.histogram(mcs, bins = 15, range = (0,29))
    pmf = distribution / len(mcs)
    return pmf

state = []
mcs_gap = []
all_pmf_simulator_mcs = []
all_pmf_flexran_mcs = []
for i in ue_speed:
    for j in scope:
        simulator_mcs = read_mcs_from_simulator(i, j, column)
        # np.savetxt("/home/ins/Downloads/MCS_from_NS3/DlMacStats_speed_yyyyy.txt", X = simulator_mcs, fmt = '%1.0f')   #check the number of MCS in txt
        pmf_simulator_mcs = calculate_pmf(simulator_mcs)
        # print(pmf_simulator_mcs)
        all_pmf_simulator_mcs.append(pmf_simulator_mcs)

        flexran_mcs = read_mcs_from_flexran(i, j)
        pmf_flexran_mcs = calculate_pmf(flexran_mcs)
        all_pmf_flexran_mcs.append(pmf_flexran_mcs)
        # print(pmf_flexran_mcs)

        mcs_sim_real_gap = pmf_flexran_mcs - pmf_simulator_mcs
        # print(mcs_sim_real_gap)
        state.append(np.array([i,j]))
        mcs_gap.append(mcs_sim_real_gap)
   
state = np.array(state)
mcs_gap = np.array(mcs_gap)
all_pmf_simulator_mcs = np.array(all_pmf_simulator_mcs)
all_pmf_flexran_mcs = np.array(all_pmf_flexran_mcs)  
# print(mcs_gap)


DIM = 2
model = DNN(input_dim=DIM, output_dim=15, seed=1111, lr=0.001, gamma=1.0, activation=torch.nn.functional.tanh)
for _ in range(100):
    loss = model.fit(state, mcs_gap)
    print(loss)

MCSgap_learned_from_bnn = model.predict(state)
# print(MCSgap_learned_from_bnn)
simulator_plus_gap = all_pmf_simulator_mcs + MCSgap_learned_from_bnn
remaining_gap_after_train = simulator_plus_gap - all_pmf_flexran_mcs

#####################################plot MCS#############################################
n = 8                       # number of file, 9 in total
x = np.arange(15)
y1 = all_pmf_simulator_mcs[n]
y2 = all_pmf_flexran_mcs[n]
y3 = simulator_plus_gap[n]
bar_width = 0.25
tabel = ["0", "2", "4", "6", "8", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28"]
plt.bar(x, y1, bar_width)
plt.bar(x+bar_width, y2, bar_width)
plt.bar(x+2*bar_width, y3, bar_width)
plt.xticks(x+bar_width, tabel)
plt.legend(labels = ["simulator","reality","simulator+gap"], fontsize = 10)
plt.xlabel('MCS', fontsize = 10)
plt.show()

##################plot heatmap (the percentage of learned gap in real gap)######################

def mean_and_reshape(mcs_gap_matrix):                # sum each line mcs_gap and reshape it in 3 *3 (9 lines in total)
    mean_mcs_gap = []
    for i in range(mcs_gap_matrix.shape[0]):
        a = np.mean(mcs_gap_matrix[i])
        mean_mcs_gap.append(a)
    mean_mcs_gap = np.array(mean_mcs_gap).reshape(3, 3)
    return mean_mcs_gap

reduced_gap_percentage = 1 - mean_and_reshape(abs(remaining_gap_after_train)) / mean_and_reshape(abs(mcs_gap))
# reduced_gap_percentage = 1- mean_and_reshape(abs(remaining_gap_after_train) / abs(mcs_gap))  # most mcs_gap equal to 0 
# print(reduced_gap_percentage)
reduced_gap_percentage = np.clip(reduced_gap_percentage, 0, 1)

# sns.set_context({"figure.figsize":(8, 8)})    #set heatmap size
sns.set(font_scale = 1.0)             #set font size in heatmap
reduced_gap_percentage = pd.DataFrame(reduced_gap_percentage)
cmap = sns.heatmap(reduced_gap_percentage, annot = True, cbar = False)      #change color with: cmap="YlGnBu", cmap="YlGnBu",
cb = cmap.figure.colorbar(cmap.collections[0])   # show the color bar
cb.ax.tick_params(labelsize = 10)      #  color bar font size

a = [0.5, 1.5, 2.5]
xlabels = ['3', '6', '10']
ylabels = ['1', '2', '3']
plt.xticks(a, xlabels, fontsize = 10)
plt.yticks(a, ylabels, fontsize = 10)
plt.xlabel('UEs scope of activity (m)', fontsize = 10)
plt.ylabel('speed of UEs (m/s)', fontsize = 10)
plt.show()
