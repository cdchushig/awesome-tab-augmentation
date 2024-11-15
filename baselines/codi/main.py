import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import argparse
import warnings
import time
from tqdm import tqdm  # For progress bars

from baselines.codi.diffusion_continuous import GaussianDiffusionTrainer, GaussianDiffusionSampler
import baselines.codi.tabular_dataload as tabular_dataload
from baselines.codi.models.tabular_unet import tabularUnet
from baselines.codi.diffusion_discrete import MultinomialDiffusion
from baselines.codi.utils import *
from utils_train import preprocess

def main(args):
    # Print training parameters for verification
    print('Training with the following parameters:')
    print(vars(args))

    # Set device and paths
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataname = args.dataname
    dataset_dir = f'data/{dataname}'
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(curr_dir, 'ckpt', dataname)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load dataset and preprocess
    with open(f'{dataset_dir}/info.json', 'r') as f:
        task_type = json.load(f)['task_type']
    
    train, train_con_data, train_dis_data, test, (transformer_con, transformer_dis, meta), con_idx, dis_idx = tabular_dataload.get_dataset(args)
    _, _, categories, d_numerical = preprocess(dataset_dir, task_type=task_type)
    num_class = np.array(categories)
    
    train_con_data = np.array(train_con_data, dtype=np.float32)
    train_con_data = torch.tensor(train_con_data, dtype=torch.float32)

    train_dis_data = np.array(train_dis_data, dtype=np.int32)
    train_dis_data = torch.tensor(train_dis_data, dtype=torch.int32)
    
    train_iter_con = infiniteloop(DataLoader(train_con_data, batch_size=args.training_batch_size))
    train_iter_dis = infiniteloop(DataLoader(train_dis_data, batch_size=args.training_batch_size))

    # Continuous Diffusion Model Setup
    args.input_size, args.output_size = train_con_data.shape[1], train_con_data.shape[1]
    args.cond_size = train_dis_data.shape[1]
    args.encoder_dim = list(map(int, args.encoder_dim_con.split(',')))
    args.nf = args.nf_con

    model_con = tabularUnet(args)
    optim_con = torch.optim.Adam(model_con.parameters(), lr=args.lr_con)
    sched_con = torch.optim.lr_scheduler.LambdaLR(optim_con, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(model_con, args.beta_1, args.beta_T, args.T).to(device)
    net_sampler = GaussianDiffusionSampler(model_con, args.beta_1, args.beta_T, args.T, args.mean_type, args.var_type).to(device)

    # Discrete Diffusion Model Setup
    args.input_size, args.output_size = train_dis_data.shape[1], train_dis_data.shape[1]
    args.cond_size = train_con_data.shape[1]
    args.encoder_dim = list(map(int, args.encoder_dim_dis.split(',')))
    args.nf = args.nf_dis

    model_dis = tabularUnet(args)
    optim_dis = torch.optim.Adam(model_dis.parameters(), lr=args.lr_dis)
    sched_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=warmup_lr)
    trainer_dis = MultinomialDiffusion(num_class, train_dis_data.shape, model_dis, args, timesteps=args.T, loss_type='vb_stochastic').to(device)

    # Print model parameter counts
    num_params_con = sum(p.numel() for p in model_con.parameters())
    num_params_dis = sum(p.numel() for p in model_dis.parameters())
    print(f'Continuous model params: {num_params_con}')
    print(f'Discrete model params: {num_params_dis}')

    # Start training using the refactored train_model function
    train_model(
        model_con=model_con, model_dis=model_dis,
        datalooper_train_con=train_iter_con, datalooper_train_dis=train_iter_dis,  # Use 'datalooper_train_con' and 'datalooper_train_dis'
        trainer=trainer, trainer_dis=trainer_dis,
        optim_con=optim_con, optim_dis=optim_dis,
        sched_con=sched_con, sched_dis=sched_dis,
        device=device, args=args, ckpt_dir=ckpt_dir,
        categories=categories, train=train, early_stopping_patience=500,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for continuous and discrete models')

    # Arguments
    parser.add_argument('--dataname', type=str, default='adult', help='Dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--training_batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--total_epochs_both', type=int, default=1000, help='Total epochs')
    parser.add_argument('--sample_step', type=int, default=100, help='Sampling step')
    parser.add_argument('--beta_1', type=float, default=0.1, help='Beta 1')
    parser.add_argument('--beta_T', type=float, default=0.9, help='Beta T')
    parser.add_argument('--T', type=int, default=1000, help='Timesteps')
    parser.add_argument('--lr_con', type=float, default=1e-4, help='Learning rate for continuous model')
    parser.add_argument('--lr_dis', type=float, default=1e-4, help='Learning rate for discrete model')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--lambda_con', type=float, default=1.0, help='Lambda for continuous model')
    parser.add_argument('--lambda_dis', type=float, default=1.0, help='Lambda for discrete model')
    parser.add_argument('--mean_type', type=str, default='eps', help='Mean type for diffusion')
    parser.add_argument('--var_type', type=str, default='learned', help='Variance type for diffusion')
    parser.add_argument('--encoder_dim_con', type=str, default='64,128', help='Encoder dimensions for continuous model')
    parser.add_argument('--encoder_dim_dis', type=str, default='64,128', help='Encoder dimensions for discrete model')
    parser.add_argument('--nf_con', type=int, default=64, help='Feature size for continuous model')
    parser.add_argument('--nf_dis', type=int, default=64, help='Feature size for discrete model')

    args = parser.parse_args()
    main(args)
