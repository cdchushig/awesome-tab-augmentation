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
        
    n_cat_tokens = len(np.unique(train_df[cat_attrs]))
    cat_dim = args.cat_emb_dim * len(cat_attrs)
    num_dim = len(num_attrs)
    encoded_dim = cat_dim + num_dim
            
    # Initialize models
    synthesizer_model = MLPSynthesizer(d_in=encoded_dim, hidden_layers=args.mlp_layers, activation=args.activation,
                                       n_cat_tokens=n_cat_tokens, n_cat_emb=args.cat_emb_dim,
                                       n_classes=len(np.unique(label_torch.numpy())), embedding_learned=False)
    
    synthesizer_model.to(device)
    
    dataname = args.dataname
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/'   
    
    synthesizer_model.load_state_dict(torch.load(f'{ckpt_dir}/mlp.pth'))
    diffuser_model = torch.load(f'{ckpt_dir}/diffuser.pth')
    
    samples = generate_samples(synthesizer_model, diffuser_model, encoded_dim, label_torch, args.diff_steps, device)
    samples_decoded = decode_samples(samples, cat_dim, num_scaler, vocab_per_attr, label_encoder, cat_attrs, num_attrs, synthesizer_model=synthesizer_model, cat_emb_dim=args.cat_emb_dim)

    print("Samples generated successfully.")
    
    # Create output directory path
    output_dir = f'synthetic/{args.dataname}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'findiff.csv')
    samples_decoded.to_csv(output_path, index=False)
    
    print(f"Shape of synthetic data: {samples_decoded.shape}")
    print(f"Synthetic data saved to {output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments for key parameters
    parser.add_argument('--dataname', type=str, default='heart', help='Name of the dataset.')
    parser.add_argument('--cat_emb_dim', type=int, default=4, help='Dimension of categorical embeddings.')
    args = parser.parse_args()
    main(args)