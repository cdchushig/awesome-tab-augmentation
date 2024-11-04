import numpy as np
import pandas as pd
import json
import argparse
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances
from collections import Counter
from scipy.spatial.distance import euclidean
from sklearn.exceptions import NotFittedError
from sklearn.cluster import KMeans
import os

# Helper function to safely execute a function with error handling.
def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return f"Error in {func.__name__}: {str(e)}"

# Function for Kolmogorov-Smirnov Test (KST).
def compute_kolmogorov_smirnov(real_data, synthetic_data, numerical_columns, print_result=True):
    ks_results = {}
    for col in numerical_columns:
        try:
            if col not in real_data.columns or col not in synthetic_data.columns:
                raise ValueError(f"Column '{col}' not found in both datasets.")
            stat, p_value = ks_2samp(real_data[col].dropna(), synthetic_data[col].dropna())
            ks_results[col] = {'statistic': stat, 'p_value': p_value}
        except Exception as e:
            ks_results[col] = f"Error: {str(e)}"
    
    if print_result:
        print("\nKolmogorov-Smirnov Test Results:")
        print(ks_results)
        
    return ks_results

# Function for Total Variation Distance (TVD).
def compute_total_variation_distance(real_data, synthetic_data, categorical_columns, print_result=True):
    tvd_results = {}
    for col in categorical_columns:
        try:
            if col not in real_data.columns or col not in synthetic_data.columns:
                raise ValueError(f"Column '{col}' not found in both datasets.")
            real_counts = Counter(real_data[col].dropna())
            synthetic_counts = Counter(synthetic_data[col].dropna())
            total_real = sum(real_counts.values())
            total_synthetic = sum(synthetic_counts.values())
            real_dist = {k: v / total_real for k, v in real_counts.items()}
            synthetic_dist = {k: v / total_synthetic for k, v in synthetic_counts.items()}
            categories = set(real_dist.keys()).union(set(synthetic_dist.keys()))
            tvd = 0.5 * sum(abs(real_dist.get(cat, 0) - synthetic_dist.get(cat, 0)) for cat in categories)
            tvd_results[col] = tvd
        except Exception as e:
            tvd_results[col] = f"Error: {str(e)}"
    
    if print_result:
        print("\nTotal Variation Distance Results:")
        print(tvd_results)
        
    return tvd_results

# Function for Kullback–Leibler Divergence (KLD).
def compute_kullback_leibler_divergence(real_data, synthetic_data, categorical_columns, print_result=True):
    kld_results = {}
    for col in categorical_columns:
        try:
            if col not in real_data.columns or col not in synthetic_data.columns:
                raise ValueError(f"Column '{col}' not found in both datasets.")
            real_counts = Counter(real_data[col].dropna())
            synthetic_counts = Counter(synthetic_data[col].dropna())
            total_real = sum(real_counts.values())
            total_synthetic = sum(synthetic_counts.values())
            real_dist = np.array([real_counts.get(cat, 0) / total_real for cat in real_counts])
            synthetic_dist = np.array([synthetic_counts.get(cat, 0) / total_synthetic for cat in real_counts])
            # To avoid division by zero
            synthetic_dist = np.clip(synthetic_dist, 1e-10, 1)
            kld = 0.5 * (np.sum(real_dist * np.log(real_dist / synthetic_dist)) + np.sum(synthetic_dist * np.log(synthetic_dist / real_dist)))
            kld_results[col] = kld
        except Exception as e:
            kld_results[col] = f"Error: {str(e)}"
    
    if print_result:
        print("\nKullback-Leibler Divergence Results:")
        print(kld_results)
        
    return kld_results

# Function for Hellinger Distance (HD).
def compute_hellinger_distance(real_data, synthetic_data, categorical_columns, print_result=True):
    hd_results = {}
    for col in categorical_columns:
        try:
            if col not in real_data.columns or col not in synthetic_data.columns:
                raise ValueError(f"Column '{col}' not found in both datasets.")
            real_counts = Counter(real_data[col].dropna())
            synthetic_counts = Counter(synthetic_data[col].dropna())
            total_real = sum(real_counts.values())
            total_synthetic = sum(synthetic_counts.values())
            real_dist = np.array([real_counts.get(cat, 0) / total_real for cat in real_counts])
            synthetic_dist = np.array([synthetic_counts.get(cat, 0) / total_synthetic for cat in real_counts])
            hd = 1 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(real_dist) - np.sqrt(synthetic_dist)) ** 2))
            hd_results[col] = hd
        except Exception as e:
            hd_results[col] = f"Error: {str(e)}"
    
    if print_result:
        print("\nHellinger Distance Results:")
        print(hd_results)
        
    return hd_results

# Function for Mean Absolute Error Probability (MAEP).
def compute_mean_absolute_error_probability(real_data, synthetic_data, categorical_columns, print_result=True):
    maep_results = {}
    for col in categorical_columns:
        try:
            if col not in real_data.columns or col not in synthetic_data.columns:
                raise ValueError(f"Column '{col}' not found in both datasets.")
            real_counts = Counter(real_data[col].dropna())
            synthetic_counts = Counter(synthetic_data[col].dropna())
            total_real = sum(real_counts.values())
            total_synthetic = sum(synthetic_counts.values())
            real_dist = np.array([real_counts.get(cat, 0) / total_real for cat in real_counts])
            synthetic_dist = np.array([synthetic_counts.get(cat, 0) / total_synthetic for cat in real_counts])
            maep = np.sum(np.abs(real_dist - synthetic_dist))
            maep_results[col] = maep
        except Exception as e:
            maep_results[col] = f"Error: {str(e)}"
    
    if print_result:
        print("\nMean Absolute Error Probability Results:")
        print(maep_results)
        
    return maep_results

# Function for Pairwise Correlation Difference (PCD).
def compute_pairwise_correlation_difference(real_data, synthetic_data, columns, print_result=True):
    try:
        real_corr = real_data[columns].corr(method='pearson')
        synthetic_corr = synthetic_data[columns].corr(method='pearson')
        pcd = np.sum(np.abs(real_corr - synthetic_corr))
    except Exception as e:
        pcd = f"Error in compute_pairwise_correlation_difference: {str(e)}"
    
    if print_result:
        print("\nPairwise Correlation Difference:")
        print(pcd)
        
    return pcd

# Function for Log-Cluster Metric (LCM).
def compute_log_cluster_metric(real_data, synthetic_data, num_clusters=5, print_result=True):
    try:
        combined_data = pd.concat([real_data, synthetic_data])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(combined_data)
        real_labels = kmeans.predict(real_data)
        synthetic_labels = kmeans.predict(synthetic_data)
        real_counts = Counter(real_labels)
        synthetic_counts = Counter(synthetic_labels)
        total_real = len(real_data)
        total_combined = len(combined_data)
        c = total_real / total_combined
        lcm = np.log(1 / num_clusters * np.sum([(real_counts[i] / (real_counts[i] + synthetic_counts[i]) - c) ** 2 for i in range(num_clusters)]))
    except NotFittedError as e:
        lcm = f"Model not fitted: {str(e)}"
    except Exception as e:
        lcm = f"Error in compute_log_cluster_metric: {str(e)}"
    
    if print_result:
        print("\nLog-Cluster Metric:")
        print(lcm)
        
    return lcm

# Function to evaluate synthetic data.
def evaluate_synthetic_data(real_data, synthetic_data, numerical_columns, categorical_columns, selected_metrics=None):
    metrics = {
        "Kolmogorov-Smirnov Test": compute_kolmogorov_smirnov,
        "Total Variation Distance": compute_total_variation_distance,
        "Kullback-Leibler Divergence": compute_kullback_leibler_divergence,
        "Hellinger Distance": compute_hellinger_distance,
        "Mean Absolute Error Probability": compute_mean_absolute_error_probability,
        "Pairwise Correlation Difference": compute_pairwise_correlation_difference,
    }

    results = {}
    for metric_name, metric_function in metrics.items():
        if selected_metrics is None or metric_name in selected_metrics:
            results[metric_name] = safe_execute(metric_function, real_data, synthetic_data, numerical_columns if 'Correlation' in metric_name else categorical_columns, print_result=True)

    return results

# Main block for command-line input
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='adult', help='Name of the dataset')
    parser.add_argument('--model', type=str, default='model', help='Name of the model')
    parser.add_argument('--path', type=str, default=None, help='The file path of the synthetic data')

    args = parser.parse_args()

    dataname = args.dataname
    model = args.model

    # Determine synthetic data path
    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path
    real_path = f'synthetic/{dataname}/real.csv'
    data_dir = f'data/{dataname}'

    # Load data and metadata
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    # Ensure column names are consistent
    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))

    # Extract column indices
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    
    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    # Extract numerical and categorical data
    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]
    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]

    # Run evaluation
    results = evaluate_synthetic_data(real_data, syn_data, num_col_idx, cat_col_idx)
    
    # Save results to a save file
    os.makedirs(f'eval/metrics/{dataname}/{model}', exist_ok=True)
    with open(f'eval/metrics/{dataname}/{model}/metrics.txt', 'w') as f:
        for metric, result in results.items():
            f.write(f'{metric}:\n{result}\n\n')
