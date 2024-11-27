import numpy as np
import pandas as pd
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_train import preprocess, TabularDataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader
from scipy.stats import entropy

pd.options.mode.chained_assignment = None

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult', help='Name of the dataset')
parser.add_argument('--model', type=str, default='model', help='Name of the model')
parser.add_argument('--path', type=str, default=None, help='Path of the synthetic data file')
args = parser.parse_args()

# Add parent directory to the system path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_kld(real_data, syn_data):
    """Calculate the Kullback-Leibler Divergence (KLD) for each feature."""
    kld_values = []
    for col in range(real_data.shape[1]):
        real_dist = np.histogram(real_data[:, col], bins=10, density=True)[0] + 1e-10
        syn_dist = np.histogram(syn_data[:, col], bins=10, density=True)[0] + 1e-10
        kld_ns = np.sum(real_dist * np.log(real_dist / syn_dist))
        kld_sn = np.sum(syn_dist * np.log(syn_dist / real_dist))
        kld_sym = 0.5 * (kld_ns + kld_sn)
        kld_values.append(kld_sym)
    return np.mean(kld_values)

def calculate_hd(real_data, syn_data):
    """Calculate the Hellinger Distance (HD) for each feature."""
    hd_values = []
    for col in range(real_data.shape[1]):
        real_dist = np.histogram(real_data[:, col], bins=10, density=True)[0]
        syn_dist = np.histogram(syn_data[:, col], bins=10, density=True)[0]
        hd = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(real_dist) - np.sqrt(syn_dist)) ** 2))
        hd_values.append(hd)
    return np.mean(hd_values)

def calculate_maep(real_data, syn_data):
    """Calculate the Marginal Attribute Error Probability (MAEP)."""
    maep = []
    for col in range(real_data.shape[1]):
        real_dist = np.histogram(real_data[:, col], bins=10, density=True)[0]
        syn_dist = np.histogram(syn_data[:, col], bins=10, density=True)[0]
        maep.append(entropy(real_dist, syn_dist))
    return np.mean(maep)

def calculate_rsvr(syn_data):
    """Calculate the Rate of Repeated Sample Vectors (RSVR)."""
    unique_samples = len(np.unique(syn_data, axis=0))
    total_samples = syn_data.shape[0]
    return 1 - (unique_samples / total_samples)

def calculate_pcd(real_data, syn_data):
    """Calculate the Pairwise Correlation Difference (PCD)."""
    corr_real = np.corrcoef(real_data.T)
    corr_syn = np.corrcoef(syn_data.T)
    return np.linalg.norm(corr_real - corr_syn)

def calculate_lcm(real_data, syn_data, k=2):
    """
    Calculate the Log-Cluster Metric (LCM).

    This metric evaluates the similarity between real and synthetic data 
    based on how they cluster together. It measures the deviation of the 
    proportion of real data points in each cluster from a balanced distribution (0.5).

    Parameters:
        real_data (ndarray): Real data as a NumPy array (rows = samples, columns = features).
        syn_data (ndarray): Synthetic data as a NumPy array (rows = samples, columns = features).
        k (int, optional): Number of clusters for K-Means. Default is 5.

    Returns:
        float: The Log-Cluster Metric (LCM), where a lower value indicates 
               better mixing of real and synthetic data in clusters.
    """
    # Combine the real and synthetic data into one dataset
    combined_data = np.vstack((real_data, syn_data))

    # Perform K-Means clustering on the combined dataset
    # Assign the data points into k clusters
    kmeans = KMeans(n_clusters=k, random_state=42).fit(combined_data)

    # Count the number of real data points in each cluster
    cluster_sizes_real = np.bincount(kmeans.labels_[:real_data.shape[0]], minlength=k)

    # Count the number of synthetic data points in each cluster
    cluster_sizes_syn = np.bincount(kmeans.labels_[real_data.shape[0]:], minlength=k)

    # Compute the proportion of real data points in each cluster
    cluster_ratios = cluster_sizes_real / (cluster_sizes_real + cluster_sizes_syn)

    # Compute the mean squared deviation of the cluster ratios from 0.5 (balanced distribution)
    mean_squared_deviation = np.mean((cluster_ratios - 0.5)**2)

    # Take the natural logarithm of the mean squared deviation to get the final metric
    return np.log(mean_squared_deviation)

def main():
    # Set dataset and model paths
    dataname = args.dataname
    model = args.model
    syn_path = args.path if args.path else f'synthetic/{dataname}/{model}.csv'
    real_path = f'synthetic/{dataname}/real.csv'
    data_dir = f'data/{dataname}'

    print(f"Using synthetic data path: {syn_path}")

    # Load metadata and data files
    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)

    # Configure dataset columns
    real_data.columns = range(len(real_data.columns))
    syn_data.columns = range(len(syn_data.columns))
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    if info['task_type'] == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    # Split real data into numerical and categorical sections
    num_real_data, cat_real_data = real_data[num_col_idx], real_data[cat_col_idx]
    num_real_data_np, cat_real_data_np = num_real_data.to_numpy(), cat_real_data.to_numpy().astype('str')

    # Split synthetic data into numerical and categorical sections
    num_syn_data, cat_syn_data = syn_data[num_col_idx], syn_data[cat_col_idx]
    num_syn_data_np, cat_syn_data_np = num_syn_data.to_numpy(), cat_syn_data.to_numpy().astype('str')

    if model.startswith('great'):
        cat_syn_data_np = cat_syn_data.to_numpy().astype('str')
        
    # One-hot encode categorical data
    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)
    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()

    # Concatenate numerical and one-hot encoded categorical data
    le_real_data = pd.DataFrame(np.concatenate((num_real_data_np, cat_real_data_oh), axis=1)).astype(float)
    le_syn_data = pd.DataFrame(np.concatenate((num_syn_data_np, cat_syn_data_oh), axis=1)).astype(float)

    # Initialize dataloaders for real and synthetic datasets
    X_syn_loader, X_real_loader = GenericDataLoader(le_syn_data), GenericDataLoader(le_real_data)

    # Statistical evaluation
    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
    qual_res = {k: v for (k, v) in qual_res.items() if "naive" in k}

    # Calculate Alpha Precision and Beta Recall
    alpha_precision, beta_recall = qual_res['delta_precision_alpha_naive'], qual_res['delta_coverage_beta_naive']
    
    # Calculate other quality metrics
    kld = calculate_kld(num_real_data_np, num_syn_data_np)
    hd = calculate_hd(num_real_data_np, num_syn_data_np)
    maep = calculate_maep(num_real_data_np, num_syn_data_np)
    rsvr = calculate_rsvr(num_syn_data_np)
    pcd = calculate_pcd(num_real_data_np, num_syn_data_np)
    lcm = calculate_lcm(num_real_data_np, num_syn_data_np)

    print(f'Alpha precision: {alpha_precision:.6f}, Beta recall: {beta_recall:.6f}')
    print(f"KLD: {kld:.6f}, HD: {hd:.6f}, MAEP: {maep:.6f}, RSVR: {rsvr:.6f}, PCD: {pcd:.6f}, LCM: {lcm:.6f}")
    
    # Save results
    save_dir = f'eval/quality/{dataname}'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/{model}.txt', 'w') as f:
        f.write(f'KLD: {kld}\n')
        f.write(f'HD: {hd}\n')
        f.write(f'MAEP: {maep}\n')
        f.write(f'RSVR: {rsvr}\n')
        f.write(f'PCD: {pcd}\n')
        f.write(f'LCM: {lcm}\n')
        f.write(f'Alpha precision: {alpha_precision}\n')
        f.write(f'Beta recall: {beta_recall}\n')

if __name__ == '__main__':
    main()

# Alpha Precision assesses how accurately the synthetic data "fits" within the distribution of real data. 
# It checks whether synthetic samples are close to real data points, indicating the synthetic data's similarity 
# and authenticity in terms of the distribution characteristics.

# Beta Recall evaluates the extent to which synthetic data covers the diversity found in real daand distributionsta. 
# It checks if synthetic samples span the full range of the real data distribution, 
# reflecting the breadth of real data in terms of feature values and distributions.

