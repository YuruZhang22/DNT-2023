import pickle

res = open('/home/ins/Desktop/robustpolicy/saves/RESULTS_PHASE_2_ATTACK_topk_5.pkl','rb')
content= pickle.load(res)

obj_path = '/home/ins/Desktop/robustpolicy/result.txt' 
ft = open(obj_path, 'w') 

for n in content:
    ft.write(str(n) +'\n')
ft.close() 

# res = open('/home/ins/Desktop/test/results_of_100_samples.pickle','rb')
# content= pickle.load(res)

# obj_path = '/home/ins/Desktop/test/results_of_100_samples.txt' 
# ft = open(obj_path, 'w') 

# for n in content:
#     ft.write(str(n) +'\n')
# ft.close() 