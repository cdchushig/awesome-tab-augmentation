import os
import time
import torch
import argparse
import json
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils_train import preprocess, TabularDataset  # Adjust path if necessary

from ctgan import CTGAN

def main(args):
    data_dir = f'data/{args.dataname}'
    info_path = f'{data_dir}/info.json'
    train_dataset_path = f'{data_dir}/train.csv'

    with open(info_path, 'r') as f:
        info = json.load(f)
        
    train_df = pd.read_csv(train_dataset_path)

    target_col_idx = info['target_col_idx'][0] if isinstance(info['target_col_idx'], list) else info['target_col_idx']
    target_col = train_df.columns[target_col_idx]

    num_cols = [train_df.columns[i] for i in info['num_col_idx']]
    cat_cols = [train_df.columns[i] for i in info['cat_col_idx']]
    cat_cols.append(target_col)

    # Initialize the CTGAN model
    ctgan = CTGAN(
        embedding_dim=256,
        generator_dim=(512,512),
        discriminator_dim=(512,512),
        batch_size=100,  # Must be a multiple of `pac`
        epochs=100,
        pac=10  # Ensure `batch_size` is divisible by this value
    )
    ctgan.fit(train_df, discrete_columns=cat_cols)
    
    # Generate synthetic data
    synthetic_data = ctgan.sample(len(train_df))

    # Create output directory path
    output_dir = f'synthetic/{args.dataname}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ctgan.csv')
    
    synthetic_data.to_csv(output_path, index=False)
    
    print(f"Shape of synthetic data: {synthetic_data.shape}")
    print(f"Synthetic data saved to {output_path}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WGANGP on tabular data')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--num_epochs', type=int, default=12000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()

    # Set device
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    main(args)
