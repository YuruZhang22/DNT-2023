import numpy as np
import pickle, os, time
from tqdm import tqdm
import torch
from bayesian_torch.dnn import DNN
from simulator import Simulator



DIM_STATE = 9
DIM_ACTION = 5

REQUIREMENT = 300 # ms

DIM = DIM_STATE + DIM_ACTION # state

STATES = dict(  baseline_loss   = [20,50],
                enb_antenna_gain= [0,10],
                enb_tx_power    = [20,40],
                enb_noise_figure= [0,20],
                ue_antenna_gain = [0,10],
                ue_tx_power     = [10,30],
                ue_noise_figure = [0,20],
                backhaul_delay  = [0,10],
                edge_delay      = [0,10], )

ACTIONS = dict( bandwidth_ul    = [15, 50],
                bandwidth_dl    = [15, 50],
                cpu_ratio       = [0.1, 1.0],
                backhaul_bw     = [100, 1000],
                edge_bw         = [100, 1000], )

ORDERS = dict(  baseline_loss   = 0,
                enb_antenna_gain= 1,
                enb_tx_power    = 2,
                enb_noise_figure= 3,
                ue_antenna_gain = 4,
                ue_tx_power     = 5,
                ue_noise_figure = 6,
                backhaul_delay  = 7,
                edge_delay      = 8, 
                bandwidth_ul    = 9,
                bandwidth_dl    = 10,
                cpu_ratio       = 11,
                backhaul_bw     = 12,
                edge_bw         = 13, )


def subtract_state_action(conf):
    if isinstance(conf, dict):
        states, actions = {}, {}
        for key, val in conf.items():
            if key in ACTIONS:
                actions[key] = val
            else:
                states[key] = val

        return states, actions
    else:
        raise ValueError("input action has to be dict!")

def calculate_qoe(latencies):
    return sum(latencies<REQUIREMENT)/len(latencies)

def calculate_usage(action):
    if isinstance(action, dict):
        res = []
        for key, val in action.items():
            if key in ACTIONS:
                res.append(val/ACTIONS[key][-1])
        return np.mean(res)
    else:
        raise ValueError("input action has to be dict!")


####################################################################################
####################################################################################
####################################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--program', type=str, default='main-mar.cc')
parser.add_argument('--numUEs', type=int, default=1)
args = parser.parse_args()




# load your dataset 
Train_X = []
Train_Y = []

DATASET = pickle.load(open("saves/measurement_simulator_grid_search_sim_slice_scratch-simulator.cc.pickle", "rb"))
all_states = pickle.load(open("saves/measurement_simulator_grid_search_sim_slice_all_states.pickle", "rb"))
all_actions = pickle.load(open("saves/measurement_simulator_grid_search_sim_slice_all_actions.pickle", "rb"))


##### uncomment the following to re-train the DNN ####
# all_states, all_actions = [], []

# for i in tqdm(range(len(DATASET))):
#     dataset = DATASET[i]
#     X = np.array(list(dataset['conf'].values()))
#     # make your order of state and actions
#     qoe = calculate_qoe(np.array(dataset['latency']))
#     # calculate the performance efficiency
#     Train_X.append(X)
#     Train_Y.append(qoe)

#     state, action = subtract_state_action(dataset['conf'])
#     all_states.append(state)
#     all_actions.append(action)

# Train_X = np.array(Train_X)
# Train_Y = np.array(Train_Y)

# creat a DNN to approximate the f(s,a) to performance  activation=torch.sigmoid,
# model = DNN(input_dim=DIM, activation=torch.sigmoid, lr=0.0001, gamma=0.99) # attention, inverse_y make sure positive value for training under relu activation func,  0.996 for 400, 0.9996 for 4000~6000, scheduler is good, but batch queries means time 10~16, so one more scale

# losses = []
# for _ in tqdm(range(100)):
#     loss = model.fit(Train_X, Train_Y)
#     losses.append(loss)
#     print(loss)

# print('done')

# torch.save(model, "saves/trained_model.pth")


model = torch.load("saves/trained_model.pth")

simulator = Simulator()



########## collect the optimal confs under certain amount of states #############
# state = []
optimal_acts = []
optimal_confs = []
optimal_utilities = []
for i in tqdm(range(160)):

    np.random.seed(int(time.time()*1000000)%1000000)
    state = np.random.choice(all_states) 
    acts, confs, utilities = [], [], [] # reset them

    for j in range(1000): # each state, we generate 10000 actions for choose
        np.random.seed(int(time.time()*1000000)%1000000)
        action = {}
        for key, val in ACTIONS.items():
            action[key] = int(100*(np.random.rand()*(val[1] - val[0]) + val[0]))/100

        # combine state and action
        conf = {}
        for key, val in state.items():
            conf[key] = val
        for key, val in action.items():
            conf[key] = val

        conf_vec = np.array(list(conf.values()))
        qoe = model.predict(np.array([conf_vec]))
        utility = qoe / calculate_usage(action)

        acts.append(action)
        confs.append(conf)
        utilities.append(utility)

    # find the maximum
    idx = np.argmax(utilities)

    optimal_acts.append(acts[idx])
    optimal_confs.append(confs[idx])
    optimal_utilities.append(utilities[idx])


############ take these confs to simulator and get the real qoe and utility ###########

import multiprocessing as mp
num_parallel = 16
iterations = int(len(optimal_confs)/num_parallel)

# reshape it, so that we can easily pick for parallel computing
optimal_acts = np.array(optimal_acts)
optimal_acts = np.reshape(optimal_acts,(-1, num_parallel))
optimal_confs = np.array(optimal_confs)
optimal_confs = np.reshape(optimal_confs,(-1, num_parallel))
optimal_utilities = np.array(optimal_utilities)
optimal_utilities = np.reshape(optimal_utilities,(-1, num_parallel))

RESULTS = []

for ite in range(iterations):

    # result = simulator.step(optimal_conf)

    acts = optimal_acts[ite]
    confs = optimal_confs[ite]
    utils = optimal_utilities[ite]

    pool = mp.Pool(num_parallel)
    results = pool.map(simulator.step, np.array(confs))
    pool.close()

    for i in range(num_parallel):

        real_utility = calculate_qoe(results[i]['performance']) / calculate_usage(acts[i])
        item = {"latencies": results[i]['performance'],
                "confs":confs[i],
                "esti_utility":utils[i],
                "real_utility":real_utility}

        RESULTS.append(item)

    # print(optimal_utility, actual_utility, optimal_conf)


pickle.dump(RESULTS, open("saves/baseline_performance_160_random_states.pickle", "wb" ))

print('done')




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