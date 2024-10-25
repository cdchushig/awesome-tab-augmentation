
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr
from sklearn.metrics import pairwise_distances
from collections import Counter

def compute_kolmogorov_smirnov(real_data, synthetic_data, numerical_columns):
    '''
    Computes the Kolmogorov-Smirnov Test (KST) for numerical columns.
    
    The KS test compares the distribution of a numerical column in real and synthetic data
    to determine if they come from the same distribution. It returns a KS statistic and a 
    p-value for each column.
    
    Parameters:
    - real_data: The original dataset with real data.
    - synthetic_data: The dataset with synthetic data.
    - numerical_columns: List of numerical columns to compare.

    Returns:
    - ks_results: Dictionary with KS statistic and p-value for each numerical column.
    '''
    ks_results = {}
    for col in numerical_columns:
        stat, p_value = ks_2samp(real_data[col], synthetic_data[col])
        ks_results[col] = {'statistic': stat, 'p_value': p_value}
    return ks_results

def compute_total_variation_distance(real_data, synthetic_data, categorical_columns):
    '''
    Computes the Total Variation Distance (TVD) for categorical columns.

    TVD measures the difference between two probability distributions. This function calculates 
    the TVD for each categorical column by comparing the distribution of values in the real 
    and synthetic datasets.
    
    Parameters:
    - real_data: The original dataset with real data.
    - synthetic_data: The dataset with synthetic data.
    - categorical_columns: List of categorical columns to compare.

    Returns:
    - tvd_results: Dictionary with TVD value for each categorical column.
    '''
    tvd_results = {}
    for col in categorical_columns:
        real_counts = Counter(real_data[col])
        synthetic_counts = Counter(synthetic_data[col])
        total_real = sum(real_counts.values())
        total_synthetic = sum(synthetic_counts.values())
        real_dist = {k: v / total_real for k, v in real_counts.items()}
        synthetic_dist = {k: v / total_synthetic for k, v in synthetic_counts.items()}
        categories = set(real_dist.keys()).union(set(synthetic_dist.keys()))
        tvd = 0.5 * sum(abs(real_dist.get(cat, 0) - synthetic_dist.get(cat, 0)) for cat in categories)
        tvd_results[col] = tvd
    return tvd_results

def compute_pearson_correlation(real_data, synthetic_data, numerical_columns):
    '''
    Computes the difference in Pearson correlation for numerical columns between real and synthetic data.

    The Pearson correlation is a measure of the linear relationship between two variables. This function
    compares the pairwise correlations of numerical columns in the real and synthetic data and returns 
    the total difference.

    Parameters:
    - real_data: The original dataset with real data.
    - synthetic_data: The dataset with synthetic data.
    - numerical_columns: List of numerical columns to compare.

    Returns:
    - correlation_results: Dictionary with the total difference in Pearson correlations.
    '''
    correlation_results = {}
    for col in numerical_columns:
        corr_real = real_data[numerical_columns].corr(method='pearson')
        corr_synthetic = synthetic_data[numerical_columns].corr(method='pearson')
        diff = np.abs(corr_real - corr_synthetic)
        correlation_results[col] = diff.sum().sum() / len(numerical_columns)
    return correlation_results

def compute_contingency_similarity(real_data, synthetic_data, categorical_columns):
    '''
    Computes contingency similarity for categorical columns.

    Contingency similarity measures the similarity between two contingency tables, which summarize
    the relationship between categorical variables. This function compares the contingency tables 
    from real and synthetic datasets and returns a similarity score for each categorical column.
    
    Parameters:
    - real_data: The original dataset with real data.
    - synthetic_data: The dataset with synthetic data.
    - categorical_columns: List of categorical columns to compare.

    Returns:
    - similarity_results: Dictionary with contingency similarity score for each categorical column.
    '''
    similarity_results = {}
    for col in categorical_columns:
        real_contingency = pd.crosstab(real_data[col], real_data[col])
        synthetic_contingency = pd.crosstab(synthetic_data[col], synthetic_data[col])
        similarity = 1 - np.mean(pairwise_distances(real_contingency, synthetic_contingency, metric='euclidean'))
        similarity_results[col] = similarity
    return similarity_results

def compute_categorical_numerical_correlation(real_data, synthetic_data, numerical_columns, categorical_columns, n_buckets=5):
    '''
    Computes correlation between numerical and categorical columns using bucketing.
    
    For each numerical column, this function creates buckets (groups of values) and calculates 
    contingency similarity between the buckets and categorical columns in both real and synthetic datasets.

    Parameters:
    - real_data: The original dataset with real data.
    - synthetic_data: The dataset with synthetic data.
    - numerical_columns: List of numerical columns to compare.
    - categorical_columns: List of categorical columns to compare.
    - n_buckets: Number of buckets for numerical values (default is 5).

    Returns:
    - corr_results: Dictionary with correlation similarity for each numerical-categorical column pair.
    '''
    corr_results = {}
    for num_col in numerical_columns:
        real_buckets = pd.qcut(real_data[num_col], q=n_buckets, duplicates='drop')
        synthetic_buckets = pd.qcut(synthetic_data[num_col], q=n_buckets, duplicates='drop')
        for cat_col in categorical_columns:
            real_contingency = pd.crosstab(real_buckets, real_data[cat_col])
            synthetic_contingency = pd.crosstab(synthetic_buckets, synthetic_data[cat_col])
            similarity = 1 - np.mean(pairwise_distances(real_contingency, synthetic_contingency, metric='euclidean'))
            corr_results[(num_col, cat_col)] = similarity
    return corr_results

def evaluate_synthetic_data(real_data, synthetic_data, numerical_columns, categorical_columns):
    '''
    Evaluates the quality of synthetic data by computing various statistical metrics.

    This function runs several tests and computes metrics to compare the real and synthetic datasets:
    - Kolmogorov-Smirnov Test for numerical columns
    - Total Variation Distance for categorical columns
    - Pearson correlation difference for numerical columns
    - Contingency similarity for categorical columns
    - Correlation between numerical and categorical columns using bucketing.

    Parameters:
    - real_data: The original dataset with real data.
    - synthetic_data: The dataset with synthetic data.
    - numerical_columns: List of numerical columns in the datasets.
    - categorical_columns: List of categorical columns in the datasets.

    Returns:
    - Dictionary containing all computed metrics.
    '''
    ks_results = compute_kolmogorov_smirnov(real_data, synthetic_data, numerical_columns)
    tvd_results = compute_total_variation_distance(real_data, synthetic_data, categorical_columns)
    pearson_corr_results = compute_pearson_correlation(real_data, synthetic_data, numerical_columns)
    contingency_similarity_results = compute_contingency_similarity(real_data, synthetic_data, categorical_columns)
    num_cat_corr_results = compute_categorical_numerical_correlation(real_data, synthetic_data, numerical_columns, categorical_columns)
    return {
        "Kolmogorov-Smirnov Test": ks_results,
        "Total Variation Distance": tvd_results,
        "Pearson Correlation Difference": pearson_corr_results,
        "Contingency Similarity": contingency_similarity_results,
        "Numerical-Categorical Correlation": num_cat_corr_results
    }
