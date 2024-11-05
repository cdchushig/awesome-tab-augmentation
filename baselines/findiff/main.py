import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import json

# Import custom modules (adjust import paths as necessary)
from baselines.findiff.MLPSynthesizer import MLPSynthesizer
from baselines.findiff.BaseDiffuser import BaseDiffuser
from baselines.findiff.findiff_modules import load_and_preprocess_data, create_dataloader, train_model, generate_samples, decode_samples, evaluate_samples


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    print(f"Using device: {device}")
    
    # Paths and dataset parameters
    data_dir = f'data/{args.dataname}'
    info_path = f'{data_dir}/info.json'
    train_dataset_path = f'{data_dir}/train.csv'
    
    # Load dataset information
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    # Load training data
    train_df = pd.read_csv(train_dataset_path)
    print("Data loaded successfully.")
    
    # Identify target column
    target_col_idx = info['target_col_idx'][0] if isinstance(info['target_col_idx'], list) else info['target_col_idx']
    target_col = train_df.columns[target_col_idx]
    
    # Identify categorical and numerical columns
    cat_attrs = [train_df.columns[i] for i in info['cat_col_idx']]
    num_attrs = [train_df.columns[i] for i in info['num_col_idx']]

    # Preprocess data
    loaded_data = load_and_preprocess_data(data=train_df, cat_attrs=cat_attrs, num_attrs=num_attrs, target_col=target_col)
    train_num_scaled, train_cat_scaled, label_tensor, num_scaler, vocab_per_attr, label_encoder = loaded_data

    # Create DataLoader
    dataloader, label_torch = create_dataloader(train_cat_scaled, train_num_scaled, label_tensor, args.batch_size)
    encoded_dim = train_cat_scaled.shape[1] + train_num_scaled.shape[1]
    n_cat_tokens = len(np.unique(train_cat_scaled.values.flatten()))
    
    # Initialize models
    synthesizer_model = MLPSynthesizer(d_in=encoded_dim, hidden_layers=args.mlp_layers, activation=args.activation,
                                       n_cat_tokens=n_cat_tokens, n_cat_emb=args.cat_emb_dim,
                                       n_classes=len(np.unique(label_torch.numpy())), embedding_learned=False)

    diffuser_model = BaseDiffuser(total_steps=args.diff_steps, beta_start=args.beta_start, beta_end=args.beta_end,
                                  scheduler=args.scheduler, device=device)

    # Optimizer and scheduler
    optimizer = Adam(synthesizer_model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Loss function
    loss_fn = nn.MSELoss()

    # Train the model
    train_epoch_losses = train_model(synthesizer_model, diffuser_model, dataloader, optimizer, lr_scheduler, loss_fn, args.num_epochs, device)
    print("Training completed.")

    # Generate and decode samples
    #samples = generate_samples(synthesizer_model, diffuser_model, encoded_dim, label_torch, args.diff_steps, device)
    #samples_decoded = decode_samples(samples, args.cat_dim, num_scaler, vocab_per_attr, label_encoder, cat_attrs, num_attrs)

    # Evaluate generated data
    #evaluate_samples(df, samples_decoded)

    #print("Sample generation and evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for MLPSynthesizer and BaseDiffuser")

    # Add arguments for key parameters
    parser.add_argument('--dataname', type=str, default='heart', help='Name of the dataset.')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use.')
    parser.add_argument('--cat_emb_dim', type=int, default=2, help='Dimension of categorical embeddings.')
    parser.add_argument('--mlp_layers', nargs='+', type=int, default=[1024, 1024, 1024, 1024], help='MLP layer sizes.')
    parser.add_argument('--activation', type=str, default='lrelu', help='Activation function to use.')
    parser.add_argument('--diff_steps', type=int, default=500, help='Number of diffusion steps.')
    parser.add_argument('--beta_start', type=float, default=1e-4, help='Initial beta for diffusion.')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Final beta for diffusion.')
    parser.add_argument('--scheduler', type=str, default='linear', help='Scheduler type for the diffusion process.')

    args = parser.parse_args()
    main(args)