import re
import numpy as np
import pickle, os, time
from tqdm import tqdm
import torch
import multiprocessing as mp
from bayesian_torch.dnn import DNN
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import matplotlib.pyplot as plt
from parameter import *


def split_dataset(Dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(Dataset) # shuffle it

    TRAIN_DATASET = Dataset[:3500]
    TEST_DATASET = Dataset[3500:]

    pickle.dump(TRAIN_DATASET, open("saves/train_dataset.pickle", "wb" ))
    pickle.dump(TEST_DATASET, open("saves/test_dataset.pickle", "wb" ))

    return TRAIN_DATASET, TEST_DATASET

def dict_to_array_ordered(input_conf):
    output_conf = np.zeros(len(ORDERS))
    for key, val in input_conf.items():
        idx = ORDERS[key]
        output_conf[idx] = val
    
    return output_conf

def extract_state_action(conf):
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
                clipped = np.clip(val, ACTIONS[key][0], ACTIONS[key][1])
                res.append(clipped/ACTIONS[key][-1])
        return np.mean(res)
    else:
        raise ValueError("input action has to be dict!")

def add_state_to_pbound(pbound, state):
    new = {}
    for key, val in pbound.items():
        low, high = val
        low = np.clip(low + state[key], STATES[key][0], STATES[key][1])
        high = np.clip(high + state[key], STATES[key][0], STATES[key][1])
        new[key] = (low, high)
    return new

def add_dict(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        z = {}
        for key, val in x.items(): # todo make sure they are with the same keys
            z[key] = x[key] + y[key]
    else: raise ValueError("they are not dict")

    return z

def add_dict_state(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        z = {}
        for key, val in x.items(): # todo make sure they are with the same keys
            z[key] = np.clip(x[key] + y[key], STATES[key][0], STATES[key][1])
    else: raise ValueError("they are not dict")

    return z

def build_conf(state, action):
    # combine state and action
    conf = {}
    for key, val in state.items(): 
        conf[key] = np.clip(val, STATES[key][0], STATES[key][1])
    for key, val in action.items(): 
        conf[key] = np.clip(val, ACTIONS[key][0], ACTIONS[key][1])

    return conf

def query_simualtor(sim, state, action):
    # combine state and action
    conf = {}
    for key, val in state.items(): np.clip(val, STATES[key][0], STATES[key][1])
    for key, val in action.items(): np.clip(val, ACTIONS[key][0], ACTIONS[key][1])

    # go to simulator
    result = sim.step(conf)

    utility = calculate_qoe(result['performance']) / calculate_usage(action)

    return utility


def search_optimal_action(model, state, length=100000, topk=1, seed=1111):

    dim = len(ACTIONS)
    # generate the state array
    state_vec = np.expand_dims(state_to_array(state),axis=-1)
    states = np.repeat(state_vec, length, axis=-1)
    
    # generate random actions
    actions = np.zeros((dim,length))
    for i in range(dim):
        np.random.seed(seed+i)
        actions[i] = np.random.randint(ACTIONS_BOUND[i,0]*10, ACTIONS_BOUND[i,1]*10, size=length)/10

    # concate confs, state at first
    states = np.transpose(states)
    actions = np.transpose(actions)
    confs = np.concatenate((states, actions), axis=-1)

    # predict the qoes
    qoes = model.predict(confs)
    usages = np.divide(actions, ACTIONS_BOUND[:,1]).mean(axis=-1)
    
    # calculate utilities
    utilities = np.divide(qoes, usages)


    optimal_actions, optimal_qoes, optimal_utilities = [], [], []
    for k in range(topk):
        idx = np.argmax(utilities)
        optimal_actions.append(array_to_action(actions[idx]))
        optimal_qoes.append(qoes[idx])
        optimal_utilities.append(utilities[idx])
        utilities[idx] = np.min(utilities)

    final_idx = np.random.choice(topk)
    optimal_action  = optimal_actions[final_idx]
    optimal_qoe     = optimal_qoes[final_idx]
    optimal_utility = optimal_utilities[final_idx]

    # find the maximum
    # idx = np.argmax(utilities)
    # optimal_action = array_to_action(actions[idx])
    # optimal_qoe = qoes[idx]
    # optimal_utility = utilities[idx]


    return optimal_action, optimal_qoe, optimal_utility


def normal_tranning(DataSet, epochs=500):

    Train_X, Train_Y = [], []
    for i in tqdm(range(len(DataSet))):
        dataset = DataSet[i]
        state, action = extract_state_action(dataset['conf'])
        state_array = state_to_array(state)
        action_array = action_to_array(action)
        X = np.concatenate((state_array, action_array), axis=-1)

        # make your order of state and actions
        qoe = calculate_qoe(np.array(dataset['latency']))

        # calculate the performance efficiency
        Train_X.append(X)
        Train_Y.append(qoe)

    Train_X = np.array(Train_X)
    Train_Y = np.array(Train_Y)

    # ## creat a DNN to approximate the f(s,a) to performance  activation=torch.sigmoid,
    model = DNN(input_dim=len(Train_X[0]), activation=torch.sigmoid, lr=0.0001, gamma=0.99) # attention, inverse_y make sure positive value for training under relu activation func,  0.996 for 400, 0.9996 for 4000~6000, scheduler is good, but batch queries means time 10~16, so one more scale

    losses = []
    for _ in tqdm(range(epochs)):
        loss = model.fit(Train_X, Train_Y)
        losses.append(loss)
        print(loss)

    return model


def generate_all_states_actions(path="saves/measurement_simulator_grid_search_sim_slice_main-mar.cc.pickle"):

    dataset = pickle.load(open(path, "rb"))

    all_states, all_actions  = [], []

    for data in dataset:
        state, action = extract_state_action(data['conf'])
        all_states.append(state)
        all_actions.append(action)

    pickle.dump(all_states, open("saves/measurement_simulator_grid_search_sim_slice_all_states.pickle", "wb" ))
    pickle.dump(all_actions, open("saves/measurement_simulator_grid_search_sim_slice_all_actions.pickle", "wb" ))


def generate_adverse_attack_state(model, state, pbounds, noise='learning', noise_scale=0.1, topk=1, seed=1111):

    decreases = []
    
    if noise == 'learning': 
        utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.01, dim=DIM_STATE)
        gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True, n_restarts_optimizer=5,)
        optimizer = BayesianOptimization(model=gpr, f=None, pbounds=pbounds, verbose=2,)
        
        for i in range(60):

            adverse = optimizer.suggest(utility, topk=topk)
                
            # add the attack on state
            attacked_state = add_dict_state(state, adverse)

            # get the action of the solver under attacked state
            attacked_action, _, attacked_utility = search_optimal_action(model, attacked_state, topk=topk, seed=seed)

            # register the adverse ONLY and the utility under attack
            optimizer.register(params=adverse, target=-attacked_utility) 

            # print('ite', i, 'attacked_utility', -attacked_utility)

            decreases.append(-attacked_utility)

        # optimal_attack = optimizer.max['params']

        all_targets = np.array([r['target'] for r in optimizer.res])

        optimal_attacks, optimal_utilities = [], []
        for k in range(topk):
            idx = np.argmax(all_targets)
            optimal_attacks.append(optimizer.res[idx]['params'])
            optimal_utilities.append(optimizer.res[idx]['target'])
            all_targets[idx] = np.min(all_targets)

        final_idx = np.random.choice(topk)
        optimal_attack  = optimal_attacks[final_idx]
        optimal_utility = optimal_utilities[final_idx]

    elif noise == 'random':
        length = 100
        optimal_attacks, optimal_utilities = [], []

        for _ in tqdm(range(length)):
            np.random.seed(int(time.time()*1000000)%1000000)
            adverse = noise_scale * np.random.randn(len(state)).clip(-1, 1)
            adverse = array_to_state(adverse * STATES_BOUND[:,1]) # scale to the state
            # add the attack on state
            attacked_state = add_dict_state(state, adverse)
            # get the action of the solver under attacked state
            attacked_action, _, attacked_utility = search_optimal_action(model, attacked_state, topk=topk, seed=seed)
            optimal_attacks.append(adverse)
            optimal_utilities.append(attacked_utility)

        final_idx = np.argmin(optimal_utilities)
        optimal_attack  = optimal_attacks[final_idx]
        optimal_utility = optimal_utilities[final_idx]
    else:
        raise ValueError('undefined noise type!')
        
    return optimal_attack, -optimal_utility, decreases



def adverse_tranning(model, DataSet, epochs=500):

    Train_X, Train_Y = [], []
    for i in tqdm(range(len(DataSet))):
        dataset = DataSet[i]
        state = dataset['state']
        action = dataset['action']
        state_array = state_to_array(state)
        action_array = action_to_array(action)
        X = np.concatenate((state_array, action_array), axis=-1)

        # make your order of state and actions
        utility = dataset['atk_utility'] # XXX here is the attacked utility
        usage = calculate_usage(action)
        qoe = utility * usage # calculate the qoe in an inverse way

        # calculate the performance efficiency
        Train_X.append(X)
        Train_Y.append(qoe)

    Train_X = np.array(Train_X)
    Train_Y = np.array(Train_Y)

    # ## creat a DNN to approximate the f(s,a) to performance  activation=torch.sigmoid,
    # model = DNN(input_dim=len(Train_X[0]), activation=torch.sigmoid, lr=0.0001, gamma=0.99) # attention, inverse_y make sure positive value for training under relu activation func,  0.996 for 400, 0.9996 for 4000~6000, scheduler is good, but batch queries means time 10~16, so one more scale
    
    # TODO learning rate previously already decreased, so new dataset may need , e.g., increase learning rate
    losses = []
    for _ in tqdm(range(epochs)):
        loss = model.fit(Train_X, Train_Y)
        losses.append(loss)
        print(loss)

    return model



def testing(simulator, model, test_dataset, topk=1, seed=1111):
    num_parallel = 16
    iterations = int(len(test_dataset)/num_parallel)
    test_dataset = test_dataset[:iterations*num_parallel] # actual is 384=16*24

    results = []
    all_confs = []
    for dataset in test_dataset:
        result = {}
        state, _ = extract_state_action(dataset['conf'])
        org_action, org_qoe, org_utility = search_optimal_action(model, state, topk=topk, seed=seed)
        result['action'] = org_action
        result['org_qoe'] = org_qoe
        result['org_utility'] = org_utility
        results.append(result)

        # get the real simulated utility from simulator
        conf = build_conf(state, org_action)
        all_confs.append(conf)


    # reshape it, so that we can easily pick for parallel computing
    all_confs = np.array(all_confs)
    all_confs = np.reshape(all_confs,(-1, num_parallel))

    sim_qoes = []
    for ite in tqdm(range(iterations)):

        confs = all_confs[ite]

        pool = mp.Pool(num_parallel)
        res = pool.map(simulator.step, np.array(confs))
        pool.close()

        for i in range(num_parallel):
            sim_qoes.append(calculate_qoe(res[i]['performance']))

    # save the sim utility here
    for i in range(len(results)):
        result = results[i]
        result['sim_utility'] = sim_qoes[i] / calculate_usage(result['action'])

    return results


def adverse_attack_train_dataset(model, train_dataset, pbounds, topk=1, seed=1111):
    all_curves = []
    results = []

    for ite in tqdm(range(len(train_dataset))):
        result = {}

        state, _ = extract_state_action(train_dataset[ite]['conf'])
        # state = eval_states[ite]
        result['state'] = state

        # generate the optimal action from the original policy    
        org_action, org_qoe, org_utility = search_optimal_action(model, state, topk=topk, seed=seed)
        result['action'] = org_action
        result['org_qoe'] = org_qoe
        result['org_utility'] = org_utility

        # generate the attack from BO-GP
        optimal_attack, atk_utility, decreases = generate_adverse_attack_state(model, state, pbounds, noise='learning', topk=topk, seed=seed)
        result['attack'] = optimal_attack
        result['atk_utility'] = atk_utility

        # collect attackes under individual states
        all_curves.append(np.array(decreases))
        results.append(result)
        # print(decreases)

    with open('saves/all_curves.pkl', 'wb') as file: pickle.dump(all_curves, file)

    return results


def testing_attacked(simulator, model, test_dataset, pbounds, noise='learning', noise_scale=0.1, topk=1, seed=1111):
    num_parallel = 16
    iterations = int(len(test_dataset)/num_parallel)
    test_dataset = test_dataset[:iterations*num_parallel] # actual is 384=16*24

    results = []
    all_confs = []
    for data in tqdm(test_dataset):
        result = {}

        state, _ = extract_state_action(data['conf'])
        # state = eval_states[ite]
        result['state'] = state

        # generate the optimal action from the original policy    
        org_action, org_qoe, org_utility = search_optimal_action(model, state, topk=topk, seed=seed)
        result['action'] = org_action
        result['org_qoe'] = org_qoe
        result['org_utility'] = org_utility

        # generate the attack from BO-GP
        optimal_attack, atk_utility, decreases = generate_adverse_attack_state(model, state, pbounds, noise=noise, noise_scale=noise_scale, topk=topk, seed=seed)
        result['attack'] = optimal_attack
        result['atk_utility'] = atk_utility
        results.append(result)

        attacked_state = add_dict_state(state, optimal_attack)
        # get the real simulated utility from simulator
        conf = build_conf(attacked_state, org_action)
        all_confs.append(conf)

     # reshape it, so that we can easily pick for parallel computing
    all_confs = np.array(all_confs)
    all_confs = np.reshape(all_confs,(-1, num_parallel))

    # with open('saves/RESULTS_PHASE_2_ATTACK.pkl', 'wb') as file: pickle.dump(results, file)
    # with open('saves/RESULTS_PHASE_2_ALL_CONFS.pkl', 'wb') as file: pickle.dump(all_confs, file)

    # results = pickle.load(open("saves/RESULTS_PHASE_2_ATTACK.pkl", "rb"))
    # all_confs = pickle.load(open("saves/RESULTS_PHASE_2_ALL_CONFS.pkl", "rb"))

    sim_qoes = []
    for ite in tqdm(range(iterations)):

        confs = all_confs[ite]

        pool = mp.Pool(num_parallel)
        res = pool.map(simulator.step, np.array(confs))
        pool.close()

        for i in range(num_parallel):
            sim_qoes.append(calculate_qoe(res[i]['performance']))

    # save the sim utility here
    for i in range(len(results)):
        result = results[i]
        result['sim_utility'] = sim_qoes[i] / calculate_usage(result['action'])


    return results

# with open('saves/RESULTS_PHASE_1.pkl', 'wb') as file: pickle.dump(RESULTS_PHASE_1, file)




















