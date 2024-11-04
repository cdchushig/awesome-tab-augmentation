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

warnings.filterwarnings("ignore")


def main(args):
    
    params = {
        'dataname': args.dataname,
        'gpu': args.gpu,
        'training_batch_size': args.training_batch_size,
        'total_epochs_both': args.total_epochs_both,
        'sample_step': args.sample_step,
        'beta_1': args.beta_1,
        'beta_T': args.beta_T,
        'T': args.T,
        'lr_con': args.lr_con,
        'lr_dis': args.lr_dis,
        'grad_clip': args.grad_clip,
        'lambda_con': args.lambda_con,
        'lambda_dis': args.lambda_dis,
        'mean_type': args.mean_type,
        'var_type': args.var_type,
        'encoder_dim_con': args.encoder_dim_con,
        'encoder_dim_dis': args.encoder_dim_dis,
        'nf_con': args.nf_con,
        'nf_dis': args.nf_dis
    }
    
    print('Training with the following parameters:')
    print(params)
    
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    dataname = args.dataname

    dataset_dir = f'data/{dataname}'
    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)
    task_type = info['task_type']

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    train, train_con_data, train_dis_data, test, (transformer_con, transformer_dis, meta), con_idx, dis_idx = tabular_dataload.get_dataset(args)
    _, _, categories, d_numerical = preprocess(dataset_dir, task_type=task_type)
    num_class = np.array(categories)

    train_con_data = torch.tensor(train_con_data.astype(np.float32)).float()
    train_dis_data = torch.tensor(train_dis_data.astype(np.int32)).long()

    train_iter_con = DataLoader(train_con_data, batch_size=args.training_batch_size)
    train_iter_dis = DataLoader(train_dis_data, batch_size=args.training_batch_size)
    datalooper_train_con = infiniteloop(train_iter_con)
    datalooper_train_dis = infiniteloop(train_iter_dis)

    # Continuous Diffusion Model Setup
    args.input_size = train_con_data.shape[1]
    args.cond_size = train_dis_data.shape[1]
    args.output_size = train_con_data.shape[1]
    args.encoder_dim = list(map(int, args.encoder_dim_con.split(',')))
    args.nf = args.nf_con
    model_con = tabularUnet(args)
    optim_con = torch.optim.Adam(model_con.parameters(), lr=args.lr_con)
    sched_con = torch.optim.lr_scheduler.LambdaLR(optim_con, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(model_con, args.beta_1, args.beta_T, args.T).to(device)
    net_sampler = GaussianDiffusionSampler(model_con, args.beta_1, args.beta_T, args.T, args.mean_type, args.var_type).to(device)

    # Discrete Diffusion Model Setup
    args.input_size = train_dis_data.shape[1]
    args.cond_size = train_con_data.shape[1]
    args.output_size = train_dis_data.shape[1]
    args.encoder_dim = list(map(int, args.encoder_dim_dis.split(',')))
    args.nf = args.nf_dis
    model_dis = tabularUnet(args)
    optim_dis = torch.optim.Adam(model_dis.parameters(), lr=args.lr_dis)
    sched_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=warmup_lr)
    trainer_dis = MultinomialDiffusion(num_class, train_dis_data.shape, model_dis, args, timesteps=args.T, loss_type='vb_stochastic').to(device)

    num_params_con = sum(p.numel() for p in model_con.parameters())
    num_params_dis = sum(p.numel() for p in model_dis.parameters())
    print('Continuous model params: %d' % (num_params_con))
    print('Discrete model params: %d' % (num_params_dis))

    scores_max_eval = -10
    total_steps_per_epoch = int(train.shape[0] / args.training_batch_size + 1)
    total_steps_both = args.total_epochs_both * total_steps_per_epoch
    print("Total steps: %d" % total_steps_both)
    print("Sample steps: %d" % (args.sample_step * total_steps_per_epoch))
    print(f"total epochs: {args.total_epochs_both}")
    print(f"Total steps per epoch: {total_steps_per_epoch}")   
    
    epoch = 0
    best_loss = float('inf')
    
    con_lr_track = args.lr_con
    dis_lr_track = args.lr_dis
    
    for epoch in range(args.total_epochs_both):
        model_con.train()
        model_dis.train()

        pbar = tqdm(range(total_steps_per_epoch), desc=f"Epoch {epoch + 1}/{args.total_epochs_both}", leave=False)
        epoch_con_loss = 0.0
        epoch_dis_loss = 0.0

        for step in pbar:
            x_0_con = next(datalooper_train_con).to(device).float()
            x_0_dis = next(datalooper_train_dis).to(device)

            ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
            con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(
                x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, categories, args
            )

            # Calculate total loss
            loss_con = con_loss + args.lambda_con * con_loss_ns
            loss_dis = dis_loss + args.lambda_dis * dis_loss_ns

            # Accumulate losses for the epoch
            epoch_con_loss += loss_con.item()
            epoch_dis_loss += loss_dis.item()

            # Optimizer steps
            optim_con.zero_grad()
            loss_con.backward()
            torch.nn.utils.clip_grad_norm_(model_con.parameters(), args.grad_clip)
            optim_con.step()
            sched_con.step()

            optim_dis.zero_grad()
            loss_dis.backward()
            torch.nn.utils.clip_grad_value_(trainer_dis.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(trainer_dis.parameters(), args.grad_clip)
            optim_dis.step()
            sched_dis.step()
            
        # Print average losses at the end of the epoch
        avg_con_loss = epoch_con_loss / total_steps_per_epoch
        avg_dis_loss = epoch_dis_loss / total_steps_per_epoch
        print(f"Epoch {epoch + 1}/{args.total_epochs_both} | Avg Continuous Loss: {avg_con_loss:.3f} | Avg Discrete Loss: {avg_dis_loss:.3f}")

        # Check if this is the best loss
        total_loss = avg_con_loss + avg_dis_loss
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model_con.state_dict(), f'{ckpt_dir}/model_con.pt')
            torch.save(model_dis.state_dict(), f'{ckpt_dir}/model_dis.pt')

        # Save checkpoints every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            torch.save(model_con.state_dict(), f'{ckpt_dir}/model_con_{epoch + 1}.pt')
            torch.save(model_dis.state_dict(), f'{ckpt_dir}/model_dis_{epoch + 1}.pt')

    print('Training completed successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of Tabular Diffusion Models')
    # Add your argument definitions here
    parser.add_argument('--dataname', type=str, required=True, help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--training_batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--total_epochs_both', type=int, default=10, help='Total epochs for training.')
    parser.add_argument('--sample_step', type=int, default=1, help='Sample step interval.')
    parser.add_argument('--beta_1', type=float, required=True, help='Beta 1 for diffusion.')
    parser.add_argument('--beta_T', type=float, required=True, help='Beta T for diffusion.')
    parser.add_argument('--T', type=int, required=True, help='Timesteps for diffusion.')
    parser.add_argument('--lr_con', type=float, default=1e-3, help='Learning rate for continuous model.')
    parser.add_argument('--lr_dis', type=float, default=1e-3, help='Learning rate for discrete model.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value.')
    parser.add_argument('--lambda_con', type=float, default=1.0, help='Lambda for continuous loss.')
    parser.add_argument('--lambda_dis', type=float, default=1.0, help='Lambda for discrete loss.')
    parser.add_argument('--mean_type', type=str, required=True, help='Mean type for the sampler.')
    parser.add_argument('--var_type', type=str, required=True, help='Variance type for the sampler.')
    parser.add_argument('--encoder_dim_con', type=str, required=True, help='Encoder dimensions for continuous model.')
    parser.add_argument('--encoder_dim_dis', type=str, required=True, help='Encoder dimensions for discrete model.')
    parser.add_argument('--nf_con', type=int, required=True, help='Number of features for continuous model.')
    parser.add_argument('--nf_dis', type=int, required=True, help='Number of features for discrete model.')

    args = parser.parse_args()
    main(args)
