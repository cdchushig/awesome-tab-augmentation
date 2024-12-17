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
        max_beta = best_params["max_beta"]
        min_beta = best_params["min_beta"]
        lambd = best_params["lambd"]
        vae_lr = best_params["vae_lr"]
        wd = best_params["wd"]
        batch_size = best_params["batch_size"]
        vae_epochs = best_params["vae_epochs"]
        num_epochs = best_params["num_epochs"]
        tabsyn_lr = best_params["tabsyn_lr"]
        latent_dim = best_params["latent_dim"]
    else:

        # VAE hyperparameters
        max_beta = trial.suggest_float("max_beta", 1e-3, 1e-1, log=True)
        min_beta = trial.suggest_float("min_beta", 1e-5, 1e-3, log=True)
        lambd = trial.suggest_float("lambd", 0.2, 1.0)
        
        vae_lr = trial.suggest_float("vae_lr", 1e-5, 1e-3, log=True)
        wd = trial.suggest_float("wd", 1e-7, 1e-4, log=True)
        batch_size = trial.suggest_int('batch_size', 64, 512, step=64)
        vae_epochs = trial.suggest_int('vae_epochs', 500, 3000, step=500)

        # Suggest hyperparameters
        num_epochs = trial.suggest_int('num_epochs', 2000, 12000, step=2000)
        tabsyn_lr = trial.suggest_float('tabsyn_lr', 1e-4, 1e-2, log=True)
        latent_dim = trial.suggest_int('latent_dim', 32, 128, step=32)  # MLP hidden layer width

    # Paths to scripts
    vae_script = os.path.join("gen_models", "tabsyn", "vae", "main.py")
    main_script = os.path.join("gen_models", "tabsyn", "main.py")
    sample_script = os.path.join("gen_models", "tabsyn", "sample.py")
    evaluate_script = os.path.join("eval", "eval_quality.py")
    
    # Train the VAE model
    vae_cmd = [
        "python", vae_script,
        "--dataname", dataname,
        "--max_beta", str(max_beta),
        "--min_beta", str(min_beta),
        "--lambd", str(lambd),
        "--lr", str(vae_lr),
        "--wd", str(wd),
        "--batch_size", str(batch_size),
        "--num_epochs", str(vae_epochs)
    ]
     
    vae_result = subprocess.run(vae_cmd, text=True, cwd=cwd, env=env, capture_output=True)
    if vae_result.returncode != 0:
        raise RuntimeError(f"VAE script failed with error:\n{vae_result.stderr}")

    # Train the model
    train_cmd = [
        "python", main_script,
        "--dataname", dataname,
        "--batch_size", str(batch_size),
        "--num_epochs", str(num_epochs),
        "--lr", str(tabsyn_lr),
        "--latent_dim", str(latent_dim),
    ]

    result = subprocess.run(train_cmd, text=True, cwd=cwd, env=env, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Training script failed with error:\n{result.stderr}")

    # Extract model path
    if "Model saved to " in result.stdout:
        model_path = result.stdout.split("Model saved to ")[1].strip()
        print(f"Model saved to {model_path}")
    else:
        raise RuntimeError("Model path not found in output.")

    # Generate synthetic data
    model_dir = os.path.dirname(model_path)
    save_path = os.path.join(model_dir, "synthetic", f"{dataname}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sample_cmd = [
        "python", sample_script,
        "--dataname", dataname,
        "--latent_dim", str(latent_dim),
    ]

    sample_result = subprocess.run(sample_cmd, capture_output=True, text=True, cwd=cwd, env=env)
    
    print(sample_result.stdout)
    
    if sample_result.returncode != 0:
        raise RuntimeError(f"Sampling script failed with error:\n{sample_result.stderr}")

    # Extract synthetic data path
    if "Saving sampled data to" in sample_result.stdout:
        sampled_path = sample_result.stdout.split("Saving sampled data to")[1].strip()
    else:
        raise RuntimeError("Synthetic data path not found in output.")

    # Evaluate synthetic data quality
    evaluate_cmd = [
        "python", evaluate_script,
        "--dataname", dataname,
        "--model", "tabsyn",
        "--path", sampled_path
    ]

    quality_result = subprocess.run(evaluate_cmd, capture_output=True, text=True, cwd=cwd)
    if quality_result.returncode != 0:
        raise RuntimeError(f"Evaluation script failed with error:\n{quality_result.stderr}")

    # Parse quality metrics
    try:
        alpha_precision = float(quality_result.stdout.split("Alpha precision: ")[1].split(",")[0])
        beta_recall = float(quality_result.stdout.split("Beta recall: ")[1].split(",")[0].split("\n")[0])
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
