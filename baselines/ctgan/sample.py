import os
import argparse
import torch
import pandas as pd
import json

from utils_train import compute_difference_samples, balance_dataset

def main(args):
    save_path = args.save_path
    dataname = args.dataname
    balance = args.balance
    
    dataset_dir = f'data/{dataname}'
    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)
    
    # Load the trained model
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = f'baselines/ctgan/models/{args.dataname}/model.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first.")

    ctgan = torch.load(model_path)

    # Load training data to determine number of samples
    train_dataset_path = f'data/{args.dataname}/train.csv'
    train_df = pd.read_csv(train_dataset_path)
    num_samples = len(train_df)
        
    # Define el número de muestras según el argumento balance
    if args.balance:
        print('Creating a balanced oversampled dataset')
        save_path = save_path.replace('.csv', '_balanced.csv')
        difference_samples, minority_percentage = compute_difference_samples(dataname)    
        num_samples = int(difference_samples/minority_percentage) * 2
        print(f"num_samples: {num_samples}")

    # Generate synthetic data
    syn_df = ctgan.sample(num_samples)
    
    if args.balance:
        original_train_df = pd.read_csv(f'data/{args.dataname}/train.csv')
        print(f"shape of original_train_df: {original_train_df.shape}")
        target_col_idx = info['target_col_idx'][0] if isinstance(info['target_col_idx'], list) else info['target_col_idx']
        target_col = original_train_df.columns[target_col_idx]
        syn_df = balance_dataset(original_train_df, syn_df, target_col, difference_samples)


    print(f"Synthetic data saved to {save_path}")
    # Save synthetic data
    syn_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample from a trained CTGAN model')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save synthetic data')
    parser.add_argument('--balance', type=bool, default=False, help='Balance the synthetic data')
    
    args = parser.parse_args()

    main(args)
