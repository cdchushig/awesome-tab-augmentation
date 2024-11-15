import numpy as np
import pandas as pd
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_train import preprocess, TabularDataset
from sklearn.preprocessing import OneHotEncoder
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader

pd.options.mode.chained_assignment = None

# Add parent directory to the system path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Argument parsing for dataset and model selection
parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult', help='Name of the dataset')
parser.add_argument('--model', type=str, default='model', help='Name of the model')
parser.add_argument('--path', type=str, default=None, help='Path of the synthetic data file')
args = parser.parse_args()

def main():
    # Set dataset and model paths
    dataname = args.dataname
    model = args.model
    syn_path = args.path if args.path else f'synthetic/{dataname}/{model}.csv'
    real_path = f'synthetic/{dataname}/real.csv'
    data_dir = f'data/{dataname}'

    print(f"Using synthetic data path: {syn_path}")

    # Load metadata and data files
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    # Configure dataset columns
    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    # Split real data into numerical and categorical sections
    num_real_data, cat_real_data = real_data[num_col_idx], real_data[cat_col_idx]
    num_real_data_np, cat_real_data_np = num_real_data.to_numpy(), cat_real_data.to_numpy().astype('str')

    # Display dataset shapes and column indices
    print(f"Real data shape: {real_data.shape}")
    print(f"Synthetic data shape: {syn_data.shape}")
    print(f"Numerical column indices: {num_col_idx}")
    print(f"Categorical column indices: {cat_col_idx}")
    print(f"Target column index: {target_col_idx}")

    # Split synthetic data into numerical and categorical sections
    num_syn_data, cat_syn_data = syn_data[num_col_idx], syn_data[cat_col_idx]
    num_syn_data_np, cat_syn_data_np = num_syn_data.to_numpy(), cat_syn_data.to_numpy().astype('str')

    if model.startswith('great'):
        cat_syn_data_np = cat_syn_data.to_numpy().astype('str')

    # One-hot encode categorical data
    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)
    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

    # Concatenate numerical and one-hot encoded categorical data
    le_real_data = pd.DataFrame(np.concatenate((num_real_data_np, cat_real_data_oh), axis=1)).astype(float)
    le_syn_data = pd.DataFrame(np.concatenate((num_syn_data_np, cat_syn_data_oh), axis=1)).astype(float)

    # Initialize dataloaders for real and synthetic datasets
    X_syn_loader, X_real_loader = GenericDataLoader(le_syn_data), GenericDataLoader(le_real_data)

    # Statistical evaluation
    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
    qual_res = {k: v for (k, v) in qual_res.items() if "naive" in k}

    # Calculate Alpha Precision and Beta Recall
    Alpha_Precision_all, Beta_Recall_all = qual_res['delta_precision_alpha_naive'], qual_res['delta_coverage_beta_naive']
    
    print(f'Alpha precision: {Alpha_Precision_all:.6f}, Beta recall: {Beta_Recall_all:.6f}')

    # Save evaluation results
    save_dir = f'eval/quality/{dataname}'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/{model}.txt', 'w') as f:
        f.write(f'Alpha precision: {Alpha_Precision_all}\n')
        f.write(f'Beta recall: {Beta_Recall_all}\n')

if __name__ == '__main__':
    main()

# Alpha Precision assesses how accurately the synthetic data "fits" within the distribution of real data. 
# It checks whether synthetic samples are close to real data points, indicating the synthetic data's similarity 
# and authenticity in terms of the distribution characteristics.

# Beta Recall evaluates the extent to which synthetic data covers the diversity found in real daand distributionsta. 
# It checks if synthetic samples span the full range of the real data distribution, 
# reflecting the breadth of real data in terms of feature values and distributions.

