import torch
import pandas as pd

import os

import argparse
import json

from baselines.great.models.great import GReaT
from baselines.great.models.great_utils import _array_to_dataframe

from utils_train import compute_difference_samples, balance_dataset

def main(args):

    dataname = args.dataname
    save_path = args.save_path
    balance = args.balance

    dataset_path = f'data/{dataname}/train.csv'
    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)
    train_df = pd.read_csv(dataset_path)


    curr_dir = os.path.dirname(os.path.abspath(__file__))

    great = GReaT("distilgpt2",                         
              epochs=200,                             
              save_steps=2000,                     
              logging_steps=50,                    
              experiment_dir="ckpt/adult",
              batch_size=24,
              #lr_scheduler_type="constant",        # Specify the learning rate scheduler 
              #learning_rate=5e-5                   # Set the inital learning rate
             )
    
    model_save_path = f'{curr_dir}/ckpt/{dataname}/model.pt'
    great.model.load_state_dict(torch.load(model_save_path))

    great.load_finetuned_model(f"{curr_dir}/ckpt/{dataname}/model.pt")

    df = _array_to_dataframe(train_df, columns=None)
    great._update_column_information(df)
    great._update_conditional_information(df, conditional_col=None)

    
    num_samples = info['train_num']
    
    # Define el número de muestras según el argumento balance
    if args.balance:
        print('Creating a balanced oversampled dataset')
        save_path = save_path.replace('.csv', '_balanced.csv')
        difference_samples, minority_percentage = compute_difference_samples(dataname)    
        num_samples = int(difference_samples/minority_percentage) * 2
        print(f"num_samples: {num_samples}")


    syn_df = great.sample(num_samples, k=100, device=args.device)
    
    if args.balance:
        original_train_df = pd.read_csv(f'data/{args.dataname}/train.csv')
        print(f"shape of original_train_df: {original_train_df.shape}")
        target_col_idx = info['target_col_idx'][0] if isinstance(info['target_col_idx'], list) else info['target_col_idx']
        target_col = original_train_df.columns[target_col_idx]
        syn_df = balance_dataset(original_train_df, syn_df, target_col, difference_samples)
    
    syn_df.to_csv(save_path, index = False)


    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GReaT')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--bs', type=int, default=16, help='(Maximum) batch size')
    parser.add_argument('--balance', type=bool, default=False, help='Balance the synthetic data')
    args = parser.parse_args()