import os
import subprocess
import argparse
import sys
import optuna

from datetime import datetime

# Define paths and environment setup
current_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(cwd)

env = os.environ.copy()
env["PYTHONPATH"] = cwd  # Add cwd to PYTHONPATH

# Objective function for Optuna
def objective(trial, dataname, best_params=None):
    if best_params:
        training_batch_size = best_params["training_batch_size"]
        total_epochs_both = best_params["total_epochs_both"]
        lr_con = best_params["lr_con"]
        lr_dis = best_params["lr_dis"]
        beta_1 = best_params["beta_1"]
        beta_T = best_params["beta_T"]
        T = best_params["T"]
        encoder_dim_con = best_params["encoder_dim_con"]
        encoder_dim_dis = best_params["encoder_dim_dis"]
    else:
        # Suggest hyperparameters to tune
        training_batch_size = trial.suggest_int("training_batch_size", 128, 1024, step=64)
        total_epochs_both = trial.suggest_int("total_epochs_both", 10000, 40000, step=10000)
        lr_con = trial.suggest_float("lr_con", 1e-5, 1e-3, log=True)
        lr_dis = trial.suggest_float("lr_dis", 1e-5, 1e-3, log=True)
        beta_1 = trial.suggest_float("beta_1", 0.0001, 0.001, log=True)
        beta_T = trial.suggest_float("beta_T", 0.01, 0.1, log=True)
        T = trial.suggest_int("T", 500, 2000, step=100)
        
        encoder_dim_con = trial.suggest_categorical("encoder_dim_con", ["64, 128", "128, 256", "256, 512"])
        encoder_dim_dis = trial.suggest_categorical("encoder_dim_dis", ["64, 128", "128, 256", "256, 512"])

    # Paths to scripts
    main_script = os.path.join("gen_models", "codi", "main.py")
    evaluate_script = os.path.join("eval", "eval_quality.py")
    sample_script = os.path.join("gen_models", "codi", "sample.py")

    # Train the model
    train_cmd = [
        "python", main_script,
        "--dataname", dataname,
        "--training_batch_size", str(training_batch_size),
        "--total_epochs_both", str(total_epochs_both),
        "--lr_con", str(lr_con),
        "--lr_dis", str(lr_dis),
        "--encoder_dim_con", encoder_dim_con,
        "--encoder_dim_dis", encoder_dim_dis,
        "--beta_1", str(beta_1),
        "--beta_T", str(beta_T),
    ]

    result = subprocess.run(train_cmd, text=True, cwd=cwd, env=env, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Training script failed with error:\n{result.stderr}")
    
    # Extract model directory
    if "model saved to:" in result.stdout:
        model_dir = result.stdout.split("model saved to: ")[1].strip()
        print(f"Model loaded from {model_dir}")
    else:
        raise RuntimeError("Model directory path not found in output.")

    # Generate synthetic data
    save_path = os.path.join(model_dir, "synthetic", f"{dataname}_synthetic.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sample_cmd = [
        "python", sample_script,
        "--model_dir", model_dir,
        "--save_path", save_path,
        "--dataname", dataname,
    ]

    sample_result = subprocess.run(sample_cmd, text=True, cwd=cwd, env=env, capture_output=True)
    if sample_result.returncode != 0:
        raise RuntimeError(f"Sampling script failed with error:\n{sample_result.stderr}")

    # Extract synthetic data path
    if "Saving sampled data to" in sample_result.stdout:
        synthetic_data_path = sample_result.stdout.split("Saving sampled data to")[1].strip()
        print(f"Synthetic data loaded from {synthetic_data_path}")
    else:
        raise RuntimeError("Synthetic data path not found in output.")

    # Evaluate synthetic data quality
    evaluate_cmd = [
        "python", evaluate_script,
        "--dataname", dataname,
        "--model", "codi",
        "--path", synthetic_data_path,
    ]

    quality_result = subprocess.run(evaluate_cmd, text=True, cwd=cwd, env=env, capture_output=True)
    if quality_result.returncode != 0:
        raise RuntimeError(f"Evaluation script failed with error:\n{quality_result.stderr}")
    
    print(quality_result.stdout)

    # Parse quality metrics
    try:
        alpha_precision = float(quality_result.stdout.split("Alpha precision: ")[1].split(",")[0])
        beta_recall = float(quality_result.stdout.split("Beta recall: ")[1].split("\n")[0])
        kld = float(quality_result.stdout.split("KLD: ")[1].split(",")[0])
    except (IndexError, ValueError):
        raise RuntimeError("Failed to parse quality metrics from evaluation output.")

    print(f"Alpha precision: {alpha_precision}, Beta recall: {beta_recall}, KLD: {kld}")

    # Compute the objective value
    objective_value = alpha_precision + beta_recall - 0.5 * kld
    print(f"Objective value: {objective_value}")

    return objective_value

# Main function
def main(args):
    
    date = datetime.now().strftime("%Y-%m-%d_%H-%M") 
    date_str = date.replace("-", "_")
    
    study = optuna.create_study(direction="maximize")  # Assuming higher quality is better
    study.optimize(lambda trial: objective(trial, args.dataname), n_trials=args.n_trials)
    
    # Save results
    optuna_results_dir = os.path.join(current_dir, "optuna_results")
    os.makedirs(optuna_results_dir, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(optuna_results_dir, f"trials_{date_str}.csv"))

    # Write results to a text file
    with open(os.path.join(optuna_results_dir, f"results_{date_str}.txt"), "w") as f:
        f.write(f"Best hyperparameters: {study.best_params}\n")
        f.write(f"Best score: {study.best_value}\n")

    print("Best hyperparameters:", study.best_params)
    print("Best score:", study.best_value)

    # Ejecutar con los mejores parámetros encontrados por Optuna
    print("\nEjecutando con los mejores parámetros encontrados por Optuna...")
    best_params = study.best_params
    objective(None, args.dataname, best_params=best_params)
    

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", required=True, help="Name of the dataset")
    parser.add_argument("--n_trials", type=int, default=25, help="Number of trials for Optuna")
    args = parser.parse_args()
    main(args)
