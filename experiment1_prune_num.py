import pandas as pd
import torch
import numpy as np
from run_one_model import run_one_model
import pickle
import sys
import argparse
import random


def main(prune_num, repeat):
    # Load data
    x = pd.read_csv('x.csv')
    y = pd.read_csv('y.csv')
    # Set gene names as index
    x.set_index(x.columns[0], inplace=True)
    print(x.shape, y.shape)

    # Convert to numpy arrays and transpose x
    x = x.values.T
    y = y.values
    
    # Map labels
    # label_map = {'Retrieval': 0, 'Acquisition': 1, 'Overlapping': 2, 'Other': 3,
    #              'Retrieval_con': 0, 'Acquisition_con': 1, 'Overlapping_con': 2, 'Other_con': 3}
    label_map = {'Retrieval': 0, 'Acquisition': 1, 'Overlapping': 2, 'Other': 3}
    # label_map = {'Retrieval_con': 0, 'Acquisition_con': 1, 'Overlapping_con': 2, 'Other_con': 3}
    # 找到y中在label_map中的的位置
    index = [i for i in range(y.shape[0]) if y[i][0] in label_map]

    y = y[index]
    x = x[index]
    print(index)
    y = np.array([label_map[i] for i in y.flatten()])
    print("Sample labels:", y[:10])

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Initialize arrays for storing AUC results
    test_auc = np.zeros(repeat)
    after_prune_test_auc = np.zeros(repeat)

    # Run model with pruning and capture AUC metrics
    for i in range(repeat):
        
        # for prune_num = 1, minibatch to generate after prune result
        seed = random.randint(0, 10000)
        input_score, _, test_metrics, _, after_prune_test_metrics, loss_, pruned_loss = run_one_model(seed, x, y, device, prune_num)
        
        # input_score, _, test_metrics, _, after_prune_test_metrics = run_one_model(i, x, y, device, prune_num)
        print(f"Repeat {i+1}/{repeat}, Test AUC: {test_metrics[1]}, After Prune Test AUC: {after_prune_test_metrics[1]}")
        print("=" * 100)
        test_auc[i] = test_metrics[1]
        after_prune_test_auc[i] = after_prune_test_metrics[1]

    # Save results to a dictionary and pickle file
    auc = {f"prune_num_{prune_num}": [test_auc, after_prune_test_auc, loss_, pruned_loss]}
    file_name = f'auc_{prune_num}_experiment.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(auc, f)
    print(f"AUC results saved to {file_name}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run model with specified pruning number and repetitions.")
    parser.add_argument('--prune_num', type=int, default=10, help="Prune number for the model.")
    parser.add_argument('--repeat', type=int, default=200, help="Number of repetitions for training.")
    args = parser.parse_args()

    # Run main function with arguments
    main(args.prune_num, args.repeat)
