import pandas as pd
import matplotlib.pyplot as plt
import torch
from kan import *
from sklearn.model_selection import train_test_split
from run_one_model import run_one_model
from tqdm import tqdm
import pickle

# read data
x = pd.read_csv('x.csv')
y = pd.read_csv('y.csv')
# the first column is the gene names, make it as the index
x.set_index(x.columns[0], inplace=True)
print(x.shape, y.shape)

# make x,y to numpy
x = x.values
y = y.values

# make batch first
x = x.T
print(x.shape, y.shape)

# make y to 0,1,2,3
label_map = {'Retrieval': 0, 'Acquisition': 1, 'Overlapping': 2, 'Other': 3, 'Retrieval_con': 0, 'Acquisition_con': 1, 'Overlapping_con': 2, 'Other_con': 3}
y = np.array([label_map[i] for i in y.flatten()])
print(y[:10])

# device cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device: ", device)

auc = {}

# prune num = 10
prune_num = 10
repeat = 200
test_auc = np.zeros(repeat)
after_prune_test_auc = np.zeros(repeat)
for i in range(repeat):
    input_score,_,test_metrics,_,after_prune_test_metrics = run_one_model(i, x, y, device, prune_num)
    print(f"repeat {i+1}/{repeat}, test auc: {test_metrics[1]}, after prune test auc: {after_prune_test_metrics[1]}")
    print(f"="*100)
    test_auc[i] = test_metrics[1]
    after_prune_test_auc[i] = after_prune_test_metrics[1]

auc[f"prune_num_{prune_num}"] = [test_auc, after_prune_test_auc]

# prune num = 20
prune_num = 20
repeat = 200
test_auc = np.zeros(repeat)
after_prune_test_auc = np.zeros(repeat)

for i in range(repeat):
    input_score,_,test_metrics,_,after_prune_test_metrics = run_one_model(i, x, y, device, prune_num)
    print(f"repeat {i+1}/{repeat}, test auc: {test_metrics[1]}, after prune test auc: {after_prune_test_metrics[1]}")
    print(f"="*100)
    test_auc[i] = test_metrics[1]
    after_prune_test_auc[i] = after_prune_test_metrics[1]

auc[f"prune_num_{prune_num}"] = [test_auc, after_prune_test_auc]

# prune num = 30
prune_num = 30
repeat = 200
test_auc = np.zeros(repeat)
after_prune_test_auc = np.zeros(repeat)

for i in range(repeat):
    input_score,_,test_metrics,_,after_prune_test_metrics = run_one_model(i, x, y, device, prune_num)
    print(f"repeat {i+1}/{repeat}, test auc: {test_metrics[1]}, after prune test auc: {after_prune_test_metrics[1]}")
    print(f"="*100)
    test_auc[i] = test_metrics[1]
    after_prune_test_auc[i] = after_prune_test_metrics[1]

auc[f"prune_num_{prune_num}"] = [test_auc, after_prune_test_auc]

# prune num = 40
prune_num = 40
repeat = 200
test_auc = np.zeros(repeat)
after_prune_test_auc = np.zeros(repeat)

for i in range(repeat):
    input_score,_,test_metrics,_,after_prune_test_metrics = run_one_model(i, x, y, device, prune_num)
    print(f"repeat {i+1}/{repeat}, test auc: {test_metrics[1]}, after prune test auc: {after_prune_test_metrics[1]}")
    print(f"="*100)
    test_auc[i] = test_metrics[1]
    after_prune_test_auc[i] = after_prune_test_metrics[1]

auc[f"prune_num_{prune_num}"] = [test_auc, after_prune_test_auc]

# prune num = 50
prune_num = 50
repeat = 200
test_auc = np.zeros(repeat)
after_prune_test_auc = np.zeros(repeat)

for i in range(repeat):
    input_score,_,test_metrics,_,after_prune_test_metrics = run_one_model(i, x, y, device, prune_num)
    print(f"repeat {i+1}/{repeat}, test auc: {test_metrics[1]}, after prune test auc: {after_prune_test_metrics[1]}")
    print(f"="*100)
    test_auc[i] = test_metrics[1]
    after_prune_test_auc[i] = after_prune_test_metrics[1]

auc[f"prune_num_{prune_num}"] = [test_auc, after_prune_test_auc]

# 将auc保存到文件
with open('auc(experiment1).pkl', 'wb') as f:
    pickle.dump(auc, f)
