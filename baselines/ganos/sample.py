import torch
import argparse
import json

import numpy as np
import pandas as pd
import os
import time
from baselines.ganos.models import WGANGP
from utils_train import preprocess, TabularDataset
from baselines.ganos.helpers import get_cat_dims  # Adjust path if necessary

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def main(args):
    # Load dataset information and initialize paths
    data_dir = f'data/{args.dataname}'
    info_path = f'{data_dir}/info.json'
    train_data_path = f'{data_dir}/train.csv'
    
    # Load dataset metadata
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    train_df = pd.read_csv(train_data_path)
    
    # Identify target column and preprocess data
    target_col_idx = info['target_col_idx'][0] if isinstance(info['target_col_idx'], list) else info['target_col_idx']
    target_col = train_df.columns[target_col_idx]
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values

    # Define numerical and categorical columns
    num_cols = [train_df.columns[i] for i in info['num_col_idx']]
    cat_cols = [train_df.columns[i] for i in info['cat_col_idx']]
    cat_cols = [col for col in cat_cols if col != target_col]
    cat_dims = get_cat_dims(X_train, cat_cols)
    
    # Initialize and configure WGANGP
    gan = WGANGP(
        write_to_disk=False, 
        num_cols=num_cols, 
        cat_dims=cat_dims,
        cat_cols=cat_cols
    )

    gan.netG = torch.load(f'{args.model_dir}/models/netG/netG_iter12000.statedict')
    gan.netD = torch.load(f'{args.model_dir}/models/netD/netD_iter12000.statedict')
        
    # preprocess data
    num_prep = make_pipeline(SimpleImputer(strategy='mean'),
                            MinMaxScaler())
    cat_prep = make_pipeline(SimpleImputer(strategy='most_frequent'),
                            OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    prep = ColumnTransformer([
        ('num', num_prep, num_cols),
        ('cat', cat_prep, cat_cols)],
        remainder='drop')
    
    X_train_trans = prep.fit_transform(X_train)
    
    # Generate synthetic data
        
    X_y_fake = gan.sample(n=int(len(X_train)))
    X_synthetic, y_synthetic = np.hsplit(X_y_fake, [-2])
    y_synthetic = y_synthetic[:, 1]
        
    # Recover the original shape
    fitted_encoder = prep.named_transformers_['cat'].named_steps['onehotencoder']
    X_syn_cat = X_synthetic[:, len(num_cols):]
    X_syn_cat_orig = fitted_encoder.inverse_transform(X_syn_cat)
    
    X_synthetic_original = np.hstack([X_synthetic[:, :len(num_cols)], X_syn_cat_orig])
            
    # Convert to DataFrame and save
    syn_df = pd.DataFrame(X_synthetic_original, columns=num_cols + cat_cols)
    syn_df[target_col] = y_synthetic
    
    # reorder columns to the original order
    syn_df = syn_df[train_df.columns]
    
    # Create output directory path
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = f'synthetic/{args.dataname}/ganos/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'synthetic_data.csv')
    
    syn_df.to_csv(output_path, index=False)
    
    print(f"Shape of synthetic data: {syn_df.shape}")
    print(f"Synthetic data saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample synthetic data using WGANGP model')
    parser.add_argument('--dataname', type=str, required=True, help='Dataset name')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model directory')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index to use for sampling, -1 for CPU')
    
    args = parser.parse_args()
    
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    
    main(args)
