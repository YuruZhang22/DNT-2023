import numpy as np

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


KEY_STATE = sorted(STATES)
KEY_ACTION = sorted(ACTIONS)


def state_to_array(state):
    return np.asarray([state[key] for key in KEY_STATE])

def array_to_state(state):
    return dict(zip(KEY_STATE, state))

def action_to_array(action):
    return np.asarray([action[key] for key in KEY_ACTION])

def array_to_action(action):
    return dict(zip(KEY_ACTION, action))

STATES_BOUND = state_to_array(STATES) # get the bounds in array for better sampling
ACTIONS_BOUND = action_to_array(ACTIONS) # get the bounds in array for better sampling

# print('')


