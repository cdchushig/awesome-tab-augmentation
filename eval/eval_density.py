import numpy as np
import pandas as pd
import os 
import json

# Import metrics for quality and diagnostic reports on synthetic data
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport

import argparse

# Set up argument parsing to allow input of data name, model name, and path to synthetic data file
parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='tabsyn')
parser.add_argument('--path', type=str, default=None, help='The file path of the synthetic data')
args = parser.parse_args()

# Function to reorder columns in real and synthetic data according to the specified metadata
def reorder(real_data, syn_data, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    # Separate numerical and categorical data in real data
    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    # Concatenate numerical and categorical data for real data and reset columns
    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    # Separate numerical and categorical data in synthetic data
    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    
    # Concatenate numerical and categorical data for synthetic data and reset columns
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    # Update metadata with column mappings for re-ordered data
    metadata = info['metadata']
    columns = metadata['columns']
    metadata['columns'] = {}

    inverse_idx_mapping = info['inverse_idx_mapping']

    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]
    
    return new_real_data, new_syn_data, metadata



if __name__ == '__main__':

    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path
    real_path = f'synthetic/{dataname}/real.csv'

    data_dir = f'data/{dataname}' 
    print(syn_path)

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    save_dir = f'eval/density/{dataname}/{model}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Reset column indices for real and synthetic data
    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    # Reorder columns and update metadata according to the info
    metadata = info['metadata']
    metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}
    new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

    # Generate quality report for synthetic data
    qual_report = QualityReport()
    qual_report.generate(new_real_data, new_syn_data, metadata)
    
    # Print summary of quality report properties
    print(qual_report.get_properties())

    # Generate diagnostic report for synthetic data
    diag_report = DiagnosticReport()
    diag_report.generate(new_real_data, new_syn_data, metadata)

    # Print summary of diagnostic report properties
    print(diag_report.get_properties())

    # Extract quality scores from the report
    quality = qual_report.get_properties()
    Shape = quality['Score'][0]
    Trend = quality['Score'][1]

    # Save shape and trend scores to a text file
    with open(f'{save_dir}/quality.txt', 'w') as f:
        f.write(f'Shape Score: {Shape}\n')
        f.write(f'Trend Score: {Trend}\n')

    # Calculate average quality score
    Quality = (Shape + Trend) / 2

    # Retrieve detailed scores for shapes, trends, and coverage diagnostics
    shapes = qual_report.get_details(property_name='Column Shapes')
    trends = qual_report.get_details(property_name='Column Pair Trends')

    # Save detailed scores to CSV files
    shapes.to_csv(f'{save_dir}/shape.csv')
    trends.to_csv(f'{save_dir}/trend.csv')