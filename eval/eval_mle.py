import numpy as np
import torch 
import pandas as pd
import os 
import sys

import json
from mle.mle import get_evaluator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='real')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')
parser.add_argument('--balance', type=bool, default=False, help='Using balanced data or not')
args = parser.parse_args()

# def preprocess(train, test, info)

#     def norm_data(data, )

if __name__ == '__main__':

    dataname = args.dataname
    model = args.model
    
    if not args.path:
        if args.balance:
            train_path = f'synthetic/{dataname}/{model}_balanced.csv'
            print('Using balanced data from ', train_path)
        else:
            train_path = f'data/{dataname}/train.csv'
            print('Using original training data from ', train_path)
    else:
        train_path = args.path
        print('Using data from ', train_path)
        
    test_path = f'data/{dataname}/test.csv'

    train = pd.read_csv(train_path).to_numpy()
    test = pd.read_csv(test_path).to_numpy()

    with open(f'data/{dataname}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']

    evaluator = get_evaluator(task_type)

    if task_type == 'regression':
        best_r2_scores, best_rmse_scores = evaluator(train, test, info)
        
        overall_scores = {}
        for score_name in ['best_r2_scores', 'best_rmse_scores']:
            overall_scores[score_name] = {}
            
            scores = eval(score_name)
            for method in scores:
                name = method['name']  
                method.pop('name')
                overall_scores[score_name][name] = method 

    else:
        best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator(train, test, info)

        overall_scores = {}
        for score_name in ['best_f1_scores', 'best_weighted_scores', 'best_auroc_scores', 'best_acc_scores', 'best_avg_scores']:
            overall_scores[score_name] = {}
            
            scores = eval(score_name)
            for method in scores:
                name = method['name']  
                method.pop('name')
                overall_scores[score_name][name] = method
                print(f'{score_name} {name} {method}')

    if not os.path.exists(f'eval/mle/{dataname}'):
        os.makedirs(f'eval/mle/{dataname}')
    
    save_path = f'eval/mle/{dataname}/original.json'
    if args.balance:
        save_path = f'eval/mle/{dataname}/{model}_balanced.json'
    print('Saving scores to ', save_path)
    with open(save_path, "w") as json_file:
        json.dump(overall_scores, json_file, indent=4, separators=(", ", ": "))