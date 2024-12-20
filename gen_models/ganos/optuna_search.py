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
        num_epochs = best_params["num_epochs"]
        batch_size = best_params["batch_size"]
        lr = best_params["lr"]
        gp_weight = best_params["gp_weight"]
        d_updates_per_g = best_params["d_updates_per_g"]
    else:
        # Suggest hyperparameters to tune
        num_epochs = trial.suggest_int("num_epochs", 1000, 20000, step=100)
        batch_size = trial.suggest_int("batch_size", 32, 128, step=16)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        gp_weight = trial.suggest_float("gp_weight", 1.0, 20.0, step=1.0)
        d_updates_per_g = trial.suggest_int("d_updates_per_g", 1, 5)

    # Paths to scripts
    main_script = os.path.join("gen_models", "ganos", "main.py")
    sample_script = os.path.join("gen_models", "ganos", "sample.py")
    evaluate_script = os.path.join("eval", "eval_quality.py")

    # Train the model
    train_cmd = [
        "python", main_script,
        "--dataname", dataname,
        "--num_epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--gp_weight", str(gp_weight),
        "--d_updates_per_g", str(d_updates_per_g),
    ]

    result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Training script failed with error:\n{result.stderr}")
    
    # Extract model path
    if "Model saved to " in result.stdout:
        model_dir = result.stdout.split("Model saved to ")[1].strip()
        print(f"Model loaded from {model_dir}")
    else:
        raise RuntimeError("Model path not found in output.")

    # Generate synthetic data
    save_path = os.path.join(model_dir, "synthetic", f"{dataname}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sample_cmd = [
        "python", sample_script,
        "--model_dir", model_dir,
        "--save_path", save_path,
        "--dataname", dataname
    ]

    sample_result = subprocess.run(sample_cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if sample_result.returncode != 0:
        raise RuntimeError(f"Sampling script failed with error:\n{sample_result.stderr}")

    
    # Extract synthetic data path
    if "Synthetic data saved to " in sample_result.stdout:
        synthetic_data_path = sample_result.stdout.split("Synthetic data saved to ")[1].strip()
        print(f"Synthetic data loaded from {synthetic_data_path}")
    else:
        raise RuntimeError("Synthetic data path not found in output.")

    # Evaluate synthetic data quality
    evaluate_cmd = [
        "python", evaluate_script,
        "--dataname", dataname,
        "--model", "ganos",
        "--path", synthetic_data_path,
    ]

    quality_result = subprocess.run(evaluate_cmd, capture_output=True, text=True, cwd=cwd, env=env)
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
    
    optuna_results_dataset_path = os.path.join(current_dir, "optuna_results", f"ganos_{args.dataname}")
    os.makedirs(optuna_results_dataset_path, exist_ok=True)
    
    study = optuna.create_study(direction="maximize", study_name=f"ganos_{args.dataname}", storage=f"sqlite:///{optuna_results_dataset_path}/{date_str}.db", load_if_exists=True)
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
