import re
import numpy as np
import pickle, os, time
from tqdm import tqdm
import torch
from bayesian_torch.dnn import DNN
from simulator import Simulator
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from bayes_opt import UtilityFunction
from scipy.special import kl_div
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import multiprocessing as mp
from parameter import *
from function import *



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--program', type=str, default='main-mar.cc')
parser.add_argument('--numUEs', type=int, default=1)
args = parser.parse_args()


# STATE_OR_ACTION = False # True is for state attack, False is for action attack
# if STATE_OR_ACTION:
#     factor_state = 1
#     factor_action = 0
# else:
#     factor_state = 0
#     factor_action = 1

factor_action = 0.1

PBOUNDS = {
            # 'baseline_loss':        (-5 * factor_state,         5 * factor_state), 
            # 'enb_antenna_gain':     (-1 * factor_state,         1 * factor_state), 
            # 'enb_tx_power':         (-2 * factor_state,         2 * factor_state), 
            # 'enb_noise_figure':     (-2 * factor_state,         2 * factor_state), 
            # 'ue_antenna_gain':      (-1 * factor_state,         1 * factor_state), 
            # 'ue_tx_power':          (-2 * factor_state,         2 * factor_state), 
            # 'ue_noise_figure':      (-2 * factor_state,         2 * factor_state), 
            # 'backhaul_delay':       (-1 * factor_state,         1 * factor_state), 
            # 'edge_delay':           (-1 * factor_state,         1 * factor_state), 
            'bandwidth_ul':         (-50 * factor_action,        50 * factor_action ),
            'bandwidth_dl':         (-50 * factor_action,        50 * factor_action ),
            'cpu_ratio':            (-1 * factor_action,         1 * factor_action ),
            'backhaul_bw':          (-1000 * factor_action,      1000 * factor_action ),
            'edge_bw':              (-1000 * factor_action,      1000 * factor_action ),
}


kappa, xi = 2.5, 0.01
utility = UtilityFunction(kind="ei", kappa=kappa, xi=xi, dim=DIM)

GPR = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True, n_restarts_optimizer=5,)

optimizer = BayesianOptimization(
    model=GPR,
    f=None,
    pbounds=PBOUNDS,
    verbose=2, 
)

logger = JSONLogger(path="BayesianOptimizationLogger.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

###################################################################################

sizes = 3888


# load your dataset 
Train_X, Train_Y = [], []

DATASET = pickle.load(open("saves/measurement_simulator_grid_search_sim_slice_main-mar.cc.pickle", "rb"))

if sizes < len(DATASET): 
    DATASET = np.random.choice(DATASET,sizes, replace=False)
    pickle.dump(DATASET, open("saves/measurement_simulator_grid_search_sim_slice_"+str(sizes)+"_scratch-simulator.cc.pickle", "wb" ))

# model = train_model(DATASET, DIM)
# torch.save(model, "saves/trained_model_"+str(sizes)+".pth")

model = torch.load("saves/trained_model_"+str(sizes)+".pth")

simulator = Simulator() # simtime=60


########## collect the optimal confs under certain amount of states #############
scale = 0.0
idxx = 1222 # TODO XXX remove this eventually
orig_PREs = []
real_PREs = []

conf_org = DATASET[idxx]['conf']
state, _ = extract_state_action(conf_org)
# sample and argmax --> action, based on state
action, org_utility = search_optimal_action(model, state)

print('org_utility', org_utility)
gaps = []
for i in tqdm(range(200)):
    
    # use BO with GP to learn to attack
    attack_action = optimizer.suggest(utility)

    adv_action = {}
    for key, val in action.items():
        # adv_action[key] = val * (1 + scale * np.clip(np.random.randn(), -1, 1))
        adv_action[key] = val + attack_action[key]
        adv_action[key] = np.clip(int(adv_action[key]*100)/100, ACTIONS[key][0], ACTIONS[key][1])

    conf = build_conf(state, adv_action)
    conf_vec = dict_to_array_ordered(conf) # XXX the order matters
    pred_qoe = model.predict(np.array([conf_vec]))
    prd_utility = pred_qoe / calculate_usage(adv_action)

    optimizer.register(params=attack_action, target=org_utility - prd_utility)

    print('ite', i, 'utility decrease', org_utility - prd_utility)

    gaps.append(org_utility - prd_utility)


with open('saves/BO_GP_Attack.pkl', 'wb') as file:
    pickle.dump(optimizer, file)

optimal_action = add_dict(optimizer.max['params'], action)
optimal_conf = build_conf(state, optimal_action)
result = simulator.step(optimal_conf)
sim_utility = calculate_qoe(result['performance']) / calculate_usage(optimal_action)

plt.plot(gaps)
plt.show()
print('done')
############ take these confs to simulator and get the real qoe and utility ###########

# import multiprocessing as mp
# num_parallel = 16
# iterations = int(len(confs_adv)/num_parallel)

# # reshape it, so that we can easily pick for parallel computing
# confs_adv = np.array(confs_adv)
# confs_adv = np.reshape(confs_adv,(-1, num_parallel))

# sim_qoes = []
# for ite in tqdm(range(iterations)):

#     # result = simulator.step(optimal_conf)

#     confs = confs_adv[ite]

#     pool = mp.Pool(num_parallel)
#     results = pool.map(simulator.step, np.array(confs))
#     pool.close()

#     for i in range(num_parallel):

#         sim_qoes.append(calculate_qoe(results[i]['performance']))
        
#         # qoes[ite*num_parallel+i].append(sim_qoe)

#         # print(qoes[ite*num_parallel+i])

# orig_qoes = np.array(orig_qoes)
# prd_qoes = np.array(prd_qoes)
# sim_qoes = np.array(sim_qoes)

# # plt.scatter(list(range(160)), qoes[:,0], label='org');plt.scatter(list(range(160)),qoes[:,1], label='prd');plt.scatter(list(range(160)),qoes[:,2], label='sim');plt.legend();plt.show()

# print("prd_qoes - sim_qoes",np.mean(np.abs(prd_qoes - sim_qoes)))
# print("orig_qoes - sim_qoes", np.mean(np.abs(orig_qoes - sim_qoes)))
# print("orig_qoes - prd_qoes", np.mean(np.abs(orig_qoes - prd_qoes)))

# pickle.dump({"orig_qoes":orig_qoes, "prd_qoes":prd_qoes, "sim_qoes":sim_qoes}, open("saves/baseline_attacked_performance_160_random_states_size"+str(sizes)+"_scale"+str(scale)+".pickle", "wb" ))

# plt.hist(orig_qoes, bins=40, cumulative=True, density=True, histtype='step',label='original')
# plt.hist(sim_qoes, bins=40, cumulative=True, density=True, histtype='step',label='simulation')
# plt.hist(prd_qoes, bins=40, cumulative=True, density=True, histtype='step',label='prediction')
# plt.legend()
# plt.savefig("results/baseline_attacked_performance_160_random_states_size"+str(sizes)+"_scale"+str(scale)+".pdf", format = 'pdf', dpi=300)
# print('done')




# resource function: sum of action percentage
# percentage of each action
# mean of all averages

# train DNN to improve its accuracy



# if __name__ == "__main__": 

#     import argparse
#     parser = argparse.ArgumentParser()
#     bandwidth_ul = np.random.randint(25, 50)
#     parser.add_argument('--program', type=str, default='scratch-simulator.cc')
#     parser.add_argument('--stage', type=str, default='offline')
#     parser.add_argument('--mode', type=str, default="grid")
#     parser.add_argument('--simtime', type=int, default=30)                  # simulation time in NS3
#     parser.add_argument('--numUEs', type=int,default=1)                    # number of users, follow the trace
#     parser.add_argument('--filename', type=str, default="Stats.txt")        # the name of the file to record the latencies, which is also output to terminal and captured then
    
#     parser.add_argument('--bandwidth_ul', type=int, default=30)             # // number of PRBs, e.g., 25, 50, or 100  // # action parameters of slicing
#     parser.add_argument('--bandwidth_dl', type=int, default=50)             # // number of PRBs, e.g., 25, 50, or 100  // # action parameters of slicing
#     parser.add_argument('--backhaul_bw', type=int, default=100)             # // backhual bandwidth, 10Mbits/s // # action parameters of slicing
#     parser.add_argument('--cpu_ratio', type=float, default=1.0)             # // the allocated CPU ratio in edge server // # action parameters of slicing
#     parser.add_argument('--edge_bw', type=int, default=22300000000)         # // edge bandwidth , bits/s
    
#     parser.add_argument('--baseline_loss', type=float, default=38.57)       #  // baseline loss, as the distrance is fixed, so log attenuation model "becomes" baseline gain
#     parser.add_argument('--enb_antenna_gain', type=float, default=5.0)      # // antenna gain
#     parser.add_argument('--enb_tx_power', type=float, default=30.0)         #  // enb tx power in dB
#     parser.add_argument('--enb_noise_figure', type=float, default=5.0)      # // enb tx noise figure (gain loss by hardware) in dB
#     parser.add_argument('--ue_antenna_gain', type=float, default=5.0)       # // antenna gain
#     parser.add_argument('--ue_tx_power', type=float, default=30.0)          # // ue tx power in dB
#     parser.add_argument('--ue_noise_figure', type=float, default=9.0)       # // ue tx noise figure (gain loss by hardware) in dB
#     # parser.add_argument('--backhaul_offset', type=float, default=0)         #  // backhual bandwidth, bits/s
#     parser.add_argument('--backhaul_delay', type=float, default=0)          # // backhual delay in milliseconds
#     parser.add_argument('--edge_delay', type=int, default=0)                # // edge delay in milliseconds
    
#     parser.add_argument('--compute_time_mean_offset', type=int, default=0)             # // factor of compute time for task computation in edge server, in millisecond (currently is exp distribution)
#     parser.add_argument('--compute_time_std_offset', type=int, default=0)             # // factor of compute time for task computation in edge server, in millisecond (currently is exp distribution)
#     parser.add_argument('--loading_time_offset', type=int, default=0)             # // factor of compute time for task computation in edge server, in millisecond (currently is exp distribution)
#     parser.add_argument('--seed', type=int, default=1111)                # // seed for simulator,i.e., NS3
#     args = parser.parse_args()
#     print(args)




# from simulator import Simulator

# simulator = Simulator(
#                     program = args.program,
#                     simtime = args.simtime,
#                     numUEs = args.numUEs,
#                     filename = args.filename,
#                     bandwidth_ul = args.bandwidth_ul,
#                     bandwidth_dl = args.bandwidth_dl,
#                     # mcs_offset_ul = args.mcs_offset_ul,
#                     # mcs_offset_dl = args.mcs_offset_dl,
#                     backhaul_bw = args.backhaul_bw,
#                     cpu_ratio = args.cpu_ratio,
#                     baseline_loss = args.baseline_loss,
#                     enb_antenna_gain = args.enb_antenna_gain,
#                     enb_tx_power = args.enb_tx_power,
#                     enb_noise_figure = args.enb_noise_figure,
#                     ue_antenna_gain = args.ue_antenna_gain,
#                     ue_tx_power = args.ue_tx_power,
#                     ue_noise_figure = args.ue_noise_figure,
#                     # backhaul_offset = args.backhaul_offset,
#                     backhaul_delay = args.backhaul_delay,
#                     edge_bw = args.edge_bw,
#                     edge_delay = args.edge_delay,
#                     compute_time_mean_offset = args.compute_time_mean_offset,
#                     compute_time_std_offset = args.compute_time_std_offset,
#                     loading_time_offset = args.loading_time_offset,
#                     seed=args.seed,
#                     )


# start_time = time.time()


# results = [simulator.step(optimal_conf) for optimal_conf in optimal_confs]
            
# # print("simulation time is ", time.time() - start_time)
# # print(results)
# RESULTS = []
# for _ in results:
#     tmp = {}
#     # tmp['optimal_conf'] = optimal_confs[i]
#     tmp['latency'] = results[i]['performance']
#     RESULTS.append(tmp)

# pickle.dump(RESULTS, open("results_of_100_samples.pickle", "wb" ))