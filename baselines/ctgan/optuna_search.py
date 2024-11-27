import os
import subprocess
import argparse
import sys
import optuna

# Define paths and environment setup
current_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(cwd)

env = os.environ.copy()
env["PYTHONPATH"] = cwd  # Add cwd to PYTHONPATH

# Objective function for Optuna
def objective(trial, dataname):
    # Suggest hyperparameters to tune
    embedding_dim = trial.suggest_int('embedding_dim', 128, 512, step=64)
    generator_dim = trial.suggest_categorical('generator_dim', ["128,128", "256,256", "128,256,128"])
    discriminator_dim = trial.suggest_categorical('discriminator_dim', ["128,128", "256,256", "128,256,128"])
    batch_size = trial.suggest_int('batch_size', 64, 256, step=32)
    epochs = trial.suggest_int('epochs', 200, 2000, step=200)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    # Paths to scripts
    main_script = os.path.join("baselines", "ctgan", "main.py")
    sample_script = os.path.join("baselines", "ctgan", "sample.py")
    evaluate_script = os.path.join("eval", "eval_quality.py")

    # Train the model
    train_cmd = [
        "python", main_script,
        "--dataname", dataname,
        "--embedding_dim", str(embedding_dim),
        "--generator_dim", generator_dim,
        "--discriminator_dim", discriminator_dim,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--learning_rate", str(learning_rate)
    ]

    result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=cwd)
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
        "--model_path", model_path,
        "--save_path", save_path,
        "--dataname", dataname
    ]

    sample_result = subprocess.run(sample_cmd, capture_output=True, text=True, cwd=cwd, env=env)
    if sample_result.returncode != 0:
        raise RuntimeError(f"Sampling script failed with error:\n{sample_result.stderr}")

    # Extract synthetic data path
    if "Synthetic data saved to " in sample_result.stdout:
        sampled_path = sample_result.stdout.split("Synthetic data saved to ")[1].strip()
    else:
        raise RuntimeError("Synthetic data path not found in output.")

    # Evaluate synthetic data quality
    evaluate_cmd = [
        "python", evaluate_script,
        "--dataname", dataname,
        "--model", "ctgan",
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
    study = optuna.create_study(direction="maximize")  # Assuming higher quality is better
    study.optimize(lambda trial: objective(trial, args.dataname), n_trials=args.n_trials)

    # Print the best parameters
    print("Best hyperparameters:", study.best_params)
    print("Best score:", study.best_value)

    # Save the study results
    optuna_results_dir = os.path.join(current_dir, "optuna_results")
    os.makedirs(optuna_results_dir, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(optuna_results_dir, "trials.csv"))
    
    # write results to a text file
    with open(os.path.join(optuna_results_dir, "results.txt"), "w") as f:
        f.write(f"Best hyperparameters: {study.best_params}\n")
        f.write(f"Best score: {study.best_value}\n")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", required=True, help="Name of the dataset")
    parser.add_argument("--n_trials", type=int, default=25, help="Number of trials for Optuna")
    args = parser.parse_args()
    main(args)
