import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time

from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
from utils_train import preprocess, TabularDataset

warnings.filterwarnings('ignore')

token_bias = True


def write_params_to_file(params, file_path):
    with open(file_path, 'w') as f:
        for key, value in params.items():
            f.write(f'{key}: {value}\n')


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def main(args):
    
    datetime = time.strftime('%Y-%m-%d-%H-%M-%S')
    
    dataname = args.dataname
    data_dir = f'data/{dataname}'

    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd
    
    lr = args.lr
    wd = args.wd
    batch_size = args.batch_size
    num_layers = args.num_layers
    d_token = args.d_token
    n_head = args.n_head
    factor = args.factor
    early_stop_patience = args.early_stop_patience
    num_epochs = args.num_epochs    
    
    device =  args.device
    
    params = {
        'dataname': dataname,
        'max_beta': max_beta,
        'min_beta': min_beta,
        'lambd': lambd,
        'lr': lr,
        'wd': wd,
        'batch_size': batch_size,
        'num_layers': num_layers,
        'd_token': d_token,
        'n_head': n_head,
        'factor': factor,
        'early_stop_patience': early_stop_patience,
        'num_epochs': num_epochs,
        'device': device,
    }
    
    print(params)

    info_path = f'data/{dataname}/info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/{datetime}' 
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    write_params_to_file(params, f'{ckpt_dir}/params.txt')

    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type = info['task_type'])
    
    print(categories) # number of cats per categorical feature

    X_train_num, _ = X_num
    X_train_cat, _ = X_cat

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)


    train_data = TabularDataset(X_train_num.float(), X_train_cat)

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)
    
    if batch_size > X_train_num.shape[0]:
        batch_size = X_train_num.shape[0]

    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    model = Model_VAE(num_layers, d_numerical, categories, d_token, n_head = n_head, factor = factor, bias = True)
    model = model.to(device)

    pre_encoder = Encoder_model(num_layers, d_numerical, categories, d_token, n_head = n_head, factor = factor).to(device)
    pre_decoder = Decoder_model(num_layers, d_numerical, categories, d_token, n_head = n_head, factor = factor).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    best_val_loss = float('inf')

    beta = max_beta
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0

        curr_count = 0

        for batch_num, batch_cat in pbar:
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
        
            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl    += loss_kld.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count
        

        '''
            Evaluation
        '''
        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)

            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()    

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset counter if improvement occurs
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print("Early stopping triggered")
                    
                    break  # Stop training if patience is exceeded

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")
                
            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd


        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
        print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item() ))

    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time)/60))
    
    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        print('Successfully load and save the model!')

        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

        np.save(f'{ckpt_dir}/train_z.npy', train_z)

        print('Successfully save pretrained embeddings in disk!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Autoencoder')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Initial Beta.')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum Beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')

    # Add the following hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs.')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Patience for early stopping.')

    # Additional hyperparameters for the model
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the VAE.')
    parser.add_argument('--d_token', type=int, default=4, help='Dimension of tokens.')
    parser.add_argument('--n_head', type=int, default=1, help='Number of heads.')
    parser.add_argument('--factor', type=int, default=32, help='Factor for the model.')

    args = parser.parse_args()

    # Check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

    main(args)