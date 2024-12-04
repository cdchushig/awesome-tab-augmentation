import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_train
from tabsyn.diffusion_utils import train_model

warnings.filterwarnings('ignore')

def main(args): 
    device = args.device
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    early_stopping_patience = 100
    latent_dim = args.latent_dim

    print({'batch_size': batch_size, 'num_epochs': num_epochs, 'lr': lr})

    train_z, _, _, _, _ = get_input_train(args)
    
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = f'{curr_dir}/ckpt/{args.dataname}'
    os.makedirs(ckpt_path, exist_ok=True)

    in_dim = train_z.shape[1]
    train_z = (train_z - train_z.mean(0)) / 2

    if batch_size > len(train_z):
        batch_size = len(train_z)

    train_loader = DataLoader(train_z, batch_size=batch_size, shuffle=True, num_workers=4)
    denoise_fn = MLPDiffusion(in_dim, latent_dim).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50, verbose=True)

    print("Model with parameters:", sum(p.numel() for p in model.parameters()))
    
    train_model(model, train_loader, optimizer, scheduler, device, num_epochs, early_stopping_patience, ckpt_path)
    
    model_path = os.path.join(ckpt_path, 'model.pt')
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of TabSyn')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model_version', type=str, default='', help='datetime of the model version')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension')
    
    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}' if args.gpu != -1 and torch.cuda.is_available() else 'cpu'
    
    main(args)
