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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--program', type=str, default='main-mar.cc')
parser.add_argument('--mode', type=str, default='adverse_test')
parser.add_argument('--noise', type=str, default='learning')
parser.add_argument('--numUEs', type=int, default=1)
# parser.add_argument('--index', type=int, default=0)
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--noise_scale', type=int, default=1)
args = parser.parse_args()


assert(args.mode in ['normal_train','normal_test',
                     'adverse_train','adverse_test', 
                     'robust_train','robust_test'])

# TODO MAKE SURE the order of state and action in Bayeisan optimizer and parameters.py are XXX IDENTICAL XXX
######################################################################################################################################################################
noise_scale = args.noise_scale

PBOUNDS = {
            'baseline_loss':        (-50 * noise_scale,         50 * noise_scale), 
            'enb_antenna_gain':     (-10 * noise_scale,         10 * noise_scale), 
            'enb_tx_power':         (-20 * noise_scale,         20 * noise_scale),   #
            'enb_noise_figure':     (-20 * noise_scale,         20 * noise_scale), 
            'ue_antenna_gain':      (-10 * noise_scale,         10 * noise_scale), 
            'ue_tx_power':          (-20 * noise_scale,         20 * noise_scale),   #
            'ue_noise_figure':      (-20 * noise_scale,         20 * noise_scale), 
            'backhaul_delay':       (-10 * noise_scale,         10 * noise_scale), 
            'edge_delay':           (-10 * noise_scale,         10 * noise_scale), 
            # 'bandwidth_ul':         (-50 * factor_action,        50 * factor_action ),
            # 'bandwidth_dl':         (-50 * factor_action,        50 * factor_action ),
            # 'cpu_ratio':            (-1 * factor_action,      1 * factor_action ),
            # 'backhaul_bw':          (-1000 * factor_action,      1000 * factor_action ),
            # 'edge_bw':              (-1000 * factor_action,      1000 * factor_action ),
}


simulator = Simulator(numUEs=args.numUEs) # by default simtime=60

########################################################################################################################################
# load  dataset 


DATASET = pickle.load(open("saves/measurement_simulator_grid_search_sim_slice_main-mar.cc.pickle", "rb"))

TRAIN_DATASET, TEST_DATASET = split_dataset(DATASET, args.seed)


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################## (phase 1) normal training  ##################################################################

if args.mode == 'normal_train':
    model = normal_tranning(TRAIN_DATASET, epochs=500)
    torch.save(model, "saves/noraml_model.pth")
    print("the trained model is in saves/noraml_model.pth")
    raise ValueError('phase1: model is well trained and save, then change the mode to test to continue')
else:
    model = torch.load("saves/noraml_model.pth")

########################################################################################################################################
########################################## (phase 1) evaluate original performance  ####################################################

if args.mode == 'normal_test':
    RESULTS_PHASE_1 = testing(simulator, model, TEST_DATASET, topk=args.topk, seed=args.seed)

    with open('saves/RESULTS_PHASE_1.pkl', 'wb') as file: pickle.dump(RESULTS_PHASE_1, file)


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
############################################## (phase 2) learn to attack  ##############################################################

if args.mode == 'adverse_train':
    RESULTS_PHASE_2_ATTACK = adverse_attack_train_dataset(model, TRAIN_DATASET, PBOUNDS, topk=args.topk, seed=args.seed)
    with open('saves/RESULTS_PHASE_2_ATTACK_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+'.pkl', 'wb') as file: pickle.dump(RESULTS_PHASE_2_ATTACK, file)
    print("PHASE 2: adverse_train results in saves/RESULTS_PHASE_2_ATTACK_topk_"+str(args.topk)+'_factor_'+str(args.noise_scale)+".pkl")
    # print("Attention! you should manually combine dataset under different args.index......")
    raise ValueError('PHASE 2: adverse train is completed')
else:
    # TODO MAKRE SURE you combined datasets manually
    RESULTS_PHASE_2_ATTACK = pickle.load(open("saves/RESULTS_PHASE_2_ATTACK"+'_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+".pkl", "rb"))


########################################################################################################################################
############################################## (phase 2) evaluate attack  ##############################################################

if args.mode == 'adverse_test':
    RESULTS_PHASE_2 = testing_attacked(simulator, model, TEST_DATASET, PBOUNDS, noise=args.noise, noise_scale=args.noise_scale, topk=args.topk, seed=args.seed)

    with open('saves/RESULTS_PHASE_2'+'_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+'.pkl', 'wb') as file: pickle.dump(RESULTS_PHASE_2, file) 
    # with open('saves/RESULTS_PHASE_2'+'_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+'.pkl', 'wb') as file: pickle.dump(RESULTS_PHASE_2, file) 
    print("PHASE 2: testing attack results in saves/RESULTS_PHASE_2"+'_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+".pkl")


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
##############################################  (phase 3) adverse training using the atk_utility  ######################################

if args.mode == 'robust_train':
    robust_model = adverse_tranning(model, RESULTS_PHASE_2_ATTACK, epochs=500)
    torch.save(robust_model, "saves/trained_robust_model"+'_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+".pth")
    print("the trained robust model is in saves/trained_robust_model"+'_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+".pth")
    print("PHASE 3: robust training in saves/trained_robust_model"+'_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+".pth")
    raise ValueError('PHASE 3: robust training in saves/trained_robust_model'+'_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+'.pth')
else:
    robust_model = torch.load("saves/trained_robust_model"+'_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+".pth")

RESULTS_PHASE_3 = testing(simulator, robust_model, TEST_DATASET, topk=args.topk, seed=args.seed)

if args.mode == 'robust_test':
    with open('saves/RESULTS_PHASE_3'+'_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+'.pkl', 'wb') as file: pickle.dump(RESULTS_PHASE_3, file)
    print("PHASE 3: testing robust training in saves/trained_robust_model"+'_topk_'+str(args.topk)+'_factor_'+str(args.noise_scale)+".pth")




print('done')

