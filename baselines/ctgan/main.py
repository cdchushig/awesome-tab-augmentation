import os
import json
import pandas as pd
import argparse
import torch

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

    # Initialize the CTGAN model with provided arguments
    ctgan = CTGAN(
        embedding_dim=args.embedding_dim,
        generator_dim=tuple(map(int, args.generator_dim.split(','))),
        discriminator_dim=tuple(map(int, args.discriminator_dim.split(','))),
        batch_size=args.batch_size,
        epochs=args.epochs,
        pac=2,
        generator_lr=args.learning_rate,
        discriminator_lr=args.learning_rate,
    )
    ctgan.fit(train_df, discrete_columns=cat_cols)
        
    # Save the trained model
    output_dir = f'baselines/ctgan/models/{args.dataname}'
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(ctgan, model_path)

    print(f"Model saved to {model_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CTGAN on tabular data')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Size of the generator embedding')
    parser.add_argument('--generator_dim', type=str, default="512,512", help='Generator hidden layer dimensions')
    parser.add_argument('--discriminator_dim', type=str, default="512,512", help='Discriminator hidden layer dimensions')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--pac', type=int, default=10, help='Number of samples to pack together')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for Adam optimizer')
    args = parser.parse_args()

    main(args)
