import matplotlib.pyplot as plt
import numpy as np
import pickle



def read_from_dataset(size, scale):
    result = pickle.load(open("saves/baseline_attacked_performance_160_random_states_size"+str(size)+"_scale"+str(scale)+".pickle", "rb" ))
    gap = np.mean(np.abs(result['orig_qoes'] - result['sim_qoes'])) 
    return gap


############ sizes ###################
# sizes = [500,1000,2000,3000,3888]
# gaps_sizes = [read_from_dataset(size, 0.0) for size in sizes]






scales = [0.0,0.2,0.4,0.6]
gaps_scales = [read_from_dataset(3888, scale) for scale in scales]


plt.scatter(scales,gaps_scales)
plt.xlabel('scale of attack noise ')
plt.ylabel('average difference')
plt.savefig('results/motivation_under_attack.pdf')
plt.show()


print('done')