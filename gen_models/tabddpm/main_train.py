import os
import argparse
import json

from gen_models.tabddpm.train import train

import src


def main(args):

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    real_data_path = f'data/{dataname}'

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    info_path = f'{real_data_path}/info.json'
    
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    num_cols_idx = info['num_col_idx']
    num_numerical_features = len(num_cols_idx)

    args.train = True
    #raw_config = src.load_config(config_path)

    print('START TRAINING')
    
    
    """train(
        **raw_config['train']['main'],
        **raw_config['diffusion_params'],
        model_save_path=model_save_path,
        real_data_path=real_data_path,
        task_type=raw_config['task_type'],
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        device=device
    )"""
    
    T_dict = {
        "seed": 0,
        "normalization": "quantile",
        "num_nan_policy": "mean",
        "cat_nan_policy": None,
        "cat_min_frequency": None,
        "cat_encoding": None,
        "y_policy": "default",
    }
    
    model_params = {
        "num_classes": 2,
        "is_y_cond": False,
        "rtdl_params": {
            "d_layers": tuple(map(int, args.d_layers.split(','))),
            "dropout": 0.0,
        }
    }
    
    steps = args.steps
    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    
    task_type = args.task_type
    model_type = args.model_type
    num_timesteps = args.num_timesteps
    gaussian_loss_type = args.gaussian_loss_type
    scheduler = args.scheduler

    train(
        model_save_path=model_save_path,
        real_data_path=real_data_path,
        steps=steps,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        task_type=task_type,
        model_type=model_type,
        model_params=model_params,
        num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler,
        T_dict=T_dict,
        num_numerical_features=num_numerical_features,
        device=device,
    )
    
    print('Model saved to: ', model_save_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--dataname', type = str, default = 'adult')
    parser.add_argument('--gpu', type = int, default=0)
    
    # training main params
    parser.add_argument('--steps', type = int, default = 100000)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--weight_decay', type = float, default = 1e-6)
    parser.add_argument('--batch_size', type = int, default = 64)
    
    parser.add_argument('--task_type', type = str, default = 'binclass')
    parser.add_argument('--model_type', type = str, default = 'mlp')    
    parser.add_argument('--num_timesteps', type = int, default = 1000)
    parser.add_argument('--gaussian_loss_type', type = str, default = 'mse')
    parser.add_argument('--scheduler', type = str, default = 'linear')
    
    parser.add_argument('--d_layers', default ="1024, 2048, 2048, 1024")

    args = parser.parse_args()
    
    main(args)