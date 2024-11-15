import os
import time
import torch
import argparse
import json
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Use absolute imports
from baselines.ganos.models import WGANGP  # Adjust path if necessary
from baselines.ganos.helpers import get_cat_dims  # Adjust path if necessary
from utils_train import preprocess, TabularDataset  # Adjust path if necessary

def main(args):
    # Paths and dataset parameters
    data_dir = f'data/{args.dataname}'
    info_path = f'{data_dir}/info.json'
    train_dataset_path = f'{data_dir}/train.csv'
    
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    gp_weight = args.gp_weight
    d_updates_per_g = args.d_updates_per_g
    
    # Load dataset information
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    # Load training data
    train_df = pd.read_csv(train_dataset_path)
    
    # Identify target column
    target_col_idx = info['target_col_idx'][0] if isinstance(info['target_col_idx'], list) else info['target_col_idx']
    target_col = train_df.columns[target_col_idx]
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    y_train = y_train.values

    # Define numerical and categorical columns
    num_cols = [train_df.columns[i] for i in info['num_col_idx']]
    cat_cols = [train_df.columns[i] for i in info['cat_col_idx']]
    # remove the target column from the categorical columns
    cat_cols = [col for col in cat_cols if col != target_col]
    cat_dims = get_cat_dims(X_train, cat_cols)
    
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
                
    # Initialize GAN model with parameters
    gan = WGANGP(write_to_disk=True,
                compute_metrics_every=500, print_every=500, plot_every=1000,
                save_model_every=500,
                num_cols=num_cols, cat_dims=cat_dims,
                transformer=prep.named_transformers_['cat']['onehotencoder'],
                cat_cols=cat_cols,
                use_aux_classifier_loss=True,
                prefix=f'baselines/ganos/{args.dataname}',
                d_updates_per_g=d_updates_per_g, gp_weight=gp_weight)

    # Fit the GAN model with arguments
    gan.fit(X_train_trans, y=y_train, 
            condition=True,
            epochs=num_epochs,  
            batch_size=batch_size,
            cat_dims=cat_dims,
            netG_kwargs={
                'hidden_layer_sizes': (128, 64),
                'n_cross_layers': 1,
                'cat_activation': 'gumbel_softmax',
                'num_activation': 'none',
                'condition_num_on_cat': True, 
                'noise_dim': 30, 
                'normal_noise': False,
                'activation': 'leaky_relu',
                'reduce_cat_dim': True,
                'use_num_hidden_layer': True,
                'layer_norm': False,
            },
            netD_kwargs={
                'hidden_layer_sizes': (128, 64, 32),
                'n_cross_layers': 2,
                'embedding_dims': 'auto',
                'activation': 'leaky_relu',
                'sigmoid_activation': False,
                'noisy_num_cols': True,
                'layer_norm': True,
            }
        )
            
    X_res, y_res = gan.resample(X_train_trans, y=y_train)
    
    print("Previous data shape: ", X_train_trans.shape)
    print("Resampled data shape: ", X_res.shape)
    
    # Save synthetic data
    syn_df = pd.DataFrame(X_res)
    syn_df[target_col] = y_res
    
    syn_df.to_csv(f'{gan.prefix}/synthetic.csv', index=False)
    print(f"Synthetic data saved to {gan.prefix}/synthetic.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WGANGP on tabular data')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--num_epochs', type=int, default=12000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gp_weight', type=float, default=10.0, help='Gradient penalty weight')
    parser.add_argument('--d_updates_per_g', type=int, default=3, help='Number of discriminator updates per generator update')
    parser.add_argument('--num_cols', type=int, default=0, help='Number of columns')
    parser.add_argument('--cat_dims', type=str, default='', help='Number of categories per categorical feature')
    
    args = parser.parse_args()

    # Set device
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    main(args)
