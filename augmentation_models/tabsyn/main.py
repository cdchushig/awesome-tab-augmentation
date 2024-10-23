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

<<<<<<< HEAD
from torch.utils.tensorboard import SummaryWriter

=======
>>>>>>> 67c1c4bce1a9ddf97bbb601dcbeb8ca17626cbd9
warnings.filterwarnings('ignore')


def main(args): 
    device = args.device
<<<<<<< HEAD
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    early_stopping_patience = 500
    lr = args.lr

    params = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
    }
    
    print(params)
=======
>>>>>>> 67c1c4bce1a9ddf97bbb601dcbeb8ca17626cbd9

    train_z, _, _, ckpt_path, _ = get_input_train(args)

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1] 

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z

<<<<<<< HEAD
    if batch_size > len(train_data):
        batch_size = len(train_data)

=======

    batch_size = 4096
>>>>>>> 67c1c4bce1a9ddf97bbb601dcbeb8ca17626cbd9
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )
<<<<<<< HEAD
        
=======

    num_epochs = 10000 + 1

>>>>>>> 67c1c4bce1a9ddf97bbb601dcbeb8ca17626cbd9
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

<<<<<<< HEAD
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50, verbose=True)
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    model.train()

    log_dir = ckpt_path + '/logs'
    writer = SummaryWriter(log_dir=log_dir)

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for i, epoch in enumerate(range(num_epochs)):
=======
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
>>>>>>> 67c1c4bce1a9ddf97bbb601dcbeb8ca17626cbd9
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})
<<<<<<< HEAD
        
        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)
        
        writer.add_scalar('Loss', curr_loss, i)

        # if lr changed, print
        if optimizer.param_groups[0]['lr'] != lr:
            lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate changed to {lr}')

        if curr_loss < best_loss:
            best_loss = curr_loss
            print('Best loss: ', best_loss)
            patience = 0
            
            torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
        else:
            patience += 1
            if patience == early_stopping_patience:
                print('Early stopping')
                print('Best loss: ', best_loss)
                break

    end_time = time.time()
    print('Time: ', end_time - start_time)
    writer.close()
=======

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

    end_time = time.time()
    print('Time: ', end_time - start_time)
>>>>>>> 67c1c4bce1a9ddf97bbb601dcbeb8ca17626cbd9

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
<<<<<<< HEAD
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
=======
>>>>>>> 67c1c4bce1a9ddf97bbb601dcbeb8ca17626cbd9

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'