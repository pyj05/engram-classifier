import pandas as pd
import matplotlib.pyplot as plt
import torch
from kan import *
from sklearn.model_selection import train_test_split
from run_one_model_without_prune import run_one_model_without_prune
from tqdm import tqdm
import pickle
import numpy as np

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

input_scores = {}

# repeat 3000 times, record input_score
repeat = 3000
input_scores_3000 = []
for i in range(repeat):
    input_score, _, test_metrics = run_one_model_without_prune(i, x, y, device)
    print(f"repeat {i+1}/{repeat}, test auc: {test_metrics[1]}")
    print(f"="*100)
    input_scores_3000.append(input_score)

n = len(input_scores_3000)

# save input_scores
input_scores_3000 = np.array([input_scores_3000[i].detach().cpu().numpy() for i in range(n)])
input_scores['input_scores_3000'] = input_scores_3000


# save input_scores
with open('input_scores.pkl', 'wb') as f:
    pickle.dump(input_scores, f)
