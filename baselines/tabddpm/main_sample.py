import os
import argparse
from baselines.tabddpm.sample import sample

import src

import pandas as pd
import json

def main(args):
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    real_data_path = f'data/{dataname}'
    sample_save_path = args.save_path

    args.train = True
    
    #raw_config = src.load_config(config_path)
    
    
    info_path = f'{real_data_path}/info.json'
    
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    num_cols_idx = info['num_col_idx']
    num_numerical_features = len(num_cols_idx)

    ''' 
    Modification of configs
    '''
    print('START SAMPLING')
    
    """sample(
        num_samples=raw_config['sample']['num_samples'],
        batch_size=raw_config['sample']['batch_size'],
        disbalance=raw_config['sample'].get('disbalance', None),
        **raw_config['diffusion_params'],
        model_save_path=model_save_path,
        sample_save_path=sample_save_path,
        real_data_path=real_data_path,
        task_type=raw_config['task_type'],
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        num_numerical_features=raw_config['num_numerical_features'],
        device=device,
        ddim=args.ddim,
        steps=args.steps
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

    
    if args.num_samples is None:
        real_data = pd.read_csv(f'{real_data_path}/train.csv')
        args.num_samples = real_data.shape[0]
    
    sample(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        disbalance=None,
        model_save_path=model_save_path,
        sample_save_path=sample_save_path,
        real_data_path=real_data_path,
        model_type="mlp",
        model_params=model_params,
        T_dict=T_dict,
        num_numerical_features=num_numerical_features,
        device=device,
        ddim=args.ddim,
        steps=args.steps
    )
    
    print(f'Synthetic data saved to {sample_save_path}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type = str, default = 'adult')
    parser.add_argument('--gpu', type = int, default=0)
    parser.add_argument('--ddim', action = 'store_true', default = False, help='Whether to use ddim sampling.')
    parser.add_argument('--steps', type=int, default = 1000)
    parser.add_argument('--save_path', type = str, default=None)
    
    parser.add_argument('--num_samples', type = int, default=None)
    parser.add_argument('--batch_size', type = int, default=128)
    
    parser.add_argument('--d_layers', type = str, default='128,128')
    

    args = parser.parse_args()
    
    main(args)