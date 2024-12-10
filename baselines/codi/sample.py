import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse
import os
import json
import warnings
import time

from baselines.codi.diffusion_continuous import GaussianDiffusionTrainer, GaussianDiffusionSampler
import baselines.codi.tabular_dataload as tabular_dataload
from baselines.codi.models.tabular_unet import tabularUnet
from baselines.codi.diffusion_discrete import MultinomialDiffusion
from baselines.codi.utils import *

from utils_train import compute_difference_samples, balance_dataset, preprocess
import pickle

warnings.filterwarnings("ignore")


def recover_data(syn_num, syn_cat, info):
    target_col_idx = info['target_col_idx']
    if info['task_type'] == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    else:
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    idx_mapping = {int(key): value for key, value in info['idx_mapping'].items()}

    syn_df = pd.DataFrame()
    for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
        if i in set(num_col_idx):
            syn_df[i] = syn_num[:, idx_mapping[i]]
        elif i in set(cat_col_idx):
            syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
        else:
            syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df


def main(args):
    print('Training with the following parameters:')
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    device = args.device

    dataset_dir = f'data/{args.dataname}'
    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)
    task_type = info['task_type']

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = args.model_dir if args.model_dir != "model" else f'{curr_dir}/ckpt/{args.dataname}'

    train, train_con_data, train_dis_data, _, (transformer_con, transformer_dis, meta), con_idx, dis_idx = tabular_dataload.get_dataset(args)
    _, _, categories, _ = preprocess(dataset_dir, task_type=task_type)

    train_con_data = torch.tensor(train_con_data.astype(np.float32)).float()
    train_dis_data = torch.tensor(train_dis_data.astype(np.int32)).long()
    
    # load net_sampler
    with open(os.path.join(ckpt_dir, 'net_sampler.pkl'), 'rb') as f:
        net_sampler = pickle.load(f)
        
    with open(os.path.join(ckpt_dir, 'trainer_dis.pkl'), 'rb') as f:
        trainer_dis = pickle.load(f)
        
    args.input_size = train_dis_data.shape[1]
    args.cond_size = train_con_data.shape[1]
    args.output_size = train_dis_data.shape[1]
    args.encoder_dim = list(map(int, args.encoder_dim_dis.split(',')))
    args.nf = args.nf_dis

    save_path = args.save_path
    num_samples = train_con_data.shape[0]

    if args.balance:
        print('Creating a balanced oversampled dataset')
        save_path = save_path.replace('.csv', '_balanced.csv')
        difference_samples, minority_percentage = compute_difference_samples(args.dataname)
        num_samples = int(difference_samples / minority_percentage) * 2

    print(f"Start sampling")
    start_time = time.time()
    with torch.no_grad():
        x_T_con = torch.randn(num_samples, train_con_data.shape[1]).to(device)
        log_x_T_dis = torch.zeros((num_samples, train_dis_data.shape[1]), device=device)
        x_con, x_dis = sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, categories, args)

    x_dis = apply_activate(x_dis, transformer_dis.output_info)
    sample_con = transformer_con.inverse_transform(x_con.detach().cpu().numpy())
    sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())

    sample = pd.DataFrame()
    con_num, dis_num = 0, 0

    for i in range(len(con_idx) + len(dis_idx)):
        if i in set(con_idx):
            sample[i] = sample_con[:, con_num]
            con_num += 1
        else:
            sample[i] = sample_dis[:, dis_num]
            dis_num += 1

    syn_df = sample
    idx_name_mapping = {int(key): value for key, value in info['idx_name_mapping'].items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)

    for col in syn_df.columns:
        if syn_df[col].dtype == 'object':
            syn_df[col] = syn_df[col].astype(float)

    if args.balance:
        original_train_df = pd.read_csv(f'data/{args.dataname}/train.csv')
        target_col_idx = info['target_col_idx'][0] if isinstance(info['target_col_idx'], list) else info['target_col_idx']
        target_col = original_train_df.columns[target_col_idx]
        syn_df = balance_dataset(original_train_df, syn_df, target_col, difference_samples)

    syn_df.to_csv(save_path, index=False)
    end_time = time.time()
    print('Sampling time:', end_time - start_time)
    print('Saving sampled data to {}'.format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None, help="Directory of the model to load")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the synthetic data")
    parser.add_argument("--dataname", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number")
    
    # General Options
    parser.add_argument('--logdir', type=str, default='./codi_exp', help='log directory')
    parser.add_argument('--train', action='store_true', help='train from scratch')
    parser.add_argument('--eval', action='store_true', help='load ckpt.pt and evaluate')

    # Network Architecture
    parser.add_argument('--encoder_dim', nargs='+', type=int, help='encoder_dim')
    parser.add_argument('--encoder_dim_con', type=str, default='64,128', help='Encoder dimensions for continuous model')
    parser.add_argument('--encoder_dim_dis', type=str, default='64,128', help='Encoder dimensions for discrete model')
    parser.add_argument('--nf', type=int, help='nf')
    parser.add_argument('--nf_con', type=int, default=64, help='Feature size for continuous model')
    parser.add_argument('--nf_dis', type=int, default=64, help='Feature size for discrete model')

    parser.add_argument('--output_size', type=int, help='output_size')
    parser.add_argument('--activation', type=str, default='relu', help='activation')

    # Training
    parser.add_argument('--T', type=int, default=50, help='total diffusion steps')
    parser.add_argument('--beta_1', type=float, default=0.00001, help='start beta value')
    parser.add_argument('--beta_T', type=float, default=0.02, help='end beta value')
    parser.add_argument('--lr_con', type=float, default=2e-03, help='target learning rate')
    parser.add_argument('--lr_dis', type=float, default=2e-03, help='target learning rate')
    parser.add_argument('--total_epochs_both', type=int, default=20000, help='total training steps')
    parser.add_argument('--grad_clip', type=float, default=1., help="gradient norm clipping")
    parser.add_argument('--parallel', action='store_true', help='multi gpu training')

    # Sampling
    parser.add_argument('--sample_step', type=int, default=2000, help='frequency of sampling')

    # Continuous diffusion model
    parser.add_argument('--mean_type', type=str, default='epsilon', choices=['xprev', 'xstart', 'epsilon'], help='predict variable')
    parser.add_argument('--var_type', type=str, default='fixedsmall', choices=['fixedlarge', 'fixedsmall'], help='variance type')

    # Contrastive Learning
    parser.add_argument('--ns_method', type=int, default=0, help='negative condition method')
    parser.add_argument('--lambda_con', type=float, default=0.2, help='lambda_con')
    parser.add_argument('--lambda_dis', type=float, default=0.2, help='lambda_dis')
    ################    
    
    parser.add_argument('--eval_batch_size', type=int, default=2100, help='batch size')
    parser.add_argument('--training_batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--balance', type=bool, default=False, help='wether to create a balanced oversampled dataset to train a classifier.')

    
    args = parser.parse_args()
    main(args)
