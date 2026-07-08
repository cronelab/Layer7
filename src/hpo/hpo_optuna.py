"""
Hyperparameter Optimization for CNN2D_LSTM using Optuna
This script uses Bayesian optimization to efficiently search the hyperparameter space.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score
import optuna
from optuna.trial import Trial
import logging
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import *
import file_loader
import pn_demo_model as pn


def train_epoch(model, optimizer, criterion, train_loader, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for x, y in train_loader:
        y = y.permute(0, 2, 1)
        x = x.unsqueeze(-1)

        x = x.to(device).float()
        y = y.to(device).float()

        optimizer.zero_grad()
        pred_last = model(x)
        tgt_last = y[:, -1, :]
        loss = criterion(pred_last, tgt_last)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def validate(model, criterion, valid_loader, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in valid_loader:
            x = x.unsqueeze(-1)
            x = x.to(device).float()
            y = y.to(device).float()
            y = y.permute(0, 2, 1)
            tgt_last = y[:, -1, :]
            preds = model(x)
            val_loss += criterion(preds, tgt_last).item()

    return val_loss / len(valid_loader)


def evaluate_model(model, test_loader, device):
    """Evaluate model and return R2 score."""
    model.eval()
    all_preds, all_tgts = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            x = x.unsqueeze(-1)
            y = y.permute(0, 2, 1)
            vel_targets = y[..., 0:2]
            vel_seq = model(x)
            tgt_last = vel_targets[:, -1, :]

            all_preds.append(vel_seq.cpu().numpy())
            all_tgts.append(tgt_last.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    tgts = np.concatenate(all_tgts, axis=0)
    r2x = r2_score(tgts[:, 0], preds[:, 0])
    r2y = r2_score(tgts[:, 1], preds[:, 1])
    score = r2x + r2y
    error = np.linalg.norm(tgts - preds)
    print(np.shape(error))
    return score/2


def objective(trial: Trial, subject_num, dataloaders, device, n_epochs=100, n_repeats=1):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        dataloaders: Tuple of (train_loader, valid_loader, test_loader)
        device: torch device
        n_epochs: Number of training epochs
        n_repeats: Number of times to repeat training with same config (for variance)

    Returns:
        Average R2 score across repeats (or single run if n_repeats=1)
    """
    train_loader, valid_loader, test_loader = dataloaders

    # Create config with suggested hyperparameters
    config = Config(subject_num)
    # Suggest hyperparameters
    config.num_kernels = trial.suggest_int('num_kernels', 2, 8, step=2)
    config.spatial_kernel_size = trial.suggest_int('spatial_kernel_size', 1, 10, step=2)
    config.temporal_kernel_size = trial.suggest_int('temporal_kernel_size', 10, 50, step=10)
    config.spatial_kernel_stride = trial.suggest_int('spatial_kernel_stride', 1, 10)
    config.temporal_kernel_stride = trial.suggest_int('temporal_kernel_stride', 1, 20)
    config.trial_length = trial.suggest_int('trial_length', 50,700, step=50)
    
    # Update derived parameters
    config.xy_dim = int((32 - config.spatial_kernel_size) / config.spatial_kernel_stride) + 1
    config.in_channels = config.xy_dim**2 * config.num_kernels

    config.rnn_hidden_size = trial.suggest_categorical('rnn_hidden_size', [128, 256, 384, 512, 640, 800])
    config.rnn_layers = trial.suggest_int('rnn_layers', 2, 6)

    config.cnn_dropout = trial.suggest_float('cnn_dropout', 0.2, 0.7)
    config.rnn_dropout = trial.suggest_float('rnn_dropout', 0.1, 0.5)

    config.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    config.weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

    config.epochs = n_epochs

    # Run training multiple times if n_repeats > 1
    all_scores = []

    for repeat in range(n_repeats):
        if n_repeats > 1:
            logging.info(f"Trial {trial.number}: Run {repeat + 1}/{n_repeats}")

        # Build model
        try:
            model = CNN2D_LSTM(config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            criterion = torch.nn.MSELoss(reduction='sum')
        except Exception as e:
            logging.error(f"Error creating model: {e}")
            return float('-inf')

        # Training loop with pruning
        best_val_score = float('-inf')

        if n_repeats == 1:
            logging.info(f"Trial {trial.number}: Starting training with {n_epochs} epochs")

        for epoch in range(n_epochs):
            train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
            val_loss = validate(model, criterion, valid_loader, device)


            # Evaluate on test set periodically
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                test_score = evaluate_model(model, valid_loader, device)
                best_val_score = max(best_val_score, test_score)

                logging.info(f"Trial {trial.number}{f' Run {repeat+1}' if n_repeats > 1 else ''} - Epoch {epoch:3d}/{n_epochs}: "
                            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                            f"Test r2: {test_score:.4f} "
                            )

                # Report intermediate value for pruning (only on first repeat)
                if repeat == 0:
                    trial.report(test_score, epoch)

                    # Check if trial should be pruned
                    if trial.should_prune():
                        logging.info(f"Trial {trial.number} pruned at epoch {epoch}")
                        raise optuna.TrialPruned()

        all_scores.append(best_val_score)
        logging.info(f"Trial {trial.number}{f' Run {repeat+1}' if n_repeats > 1 else ''} completed with R2: {best_val_score:.4f}")

    # Return median score across all repeats (more robust to outliers)
    median_score = np.median(all_scores)
    if n_repeats > 1:
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        logging.info(f"Trial {trial.number} - Median R2: {median_score:.4f}, Mean: {mean_score:.4f} ± {std_score:.4f} (over {n_repeats} runs)")
        logging.info(f"  Individual scores: {[f'{s:.4f}' for s in all_scores]}")

    return median_score


def main():
    """Main function to run HPO."""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='HPO for CNN2D_LSTM on specific participant')
    parser.add_argument('--subject', type=int, default=3, choices=[2, 3, 4,5, 102005,102005411],
                        help='Subject number (2, 3, 4 or 5). Default: 3')
    parser.add_argument('--data-path', type=str, default='D:\\Becca\\Data\\PN_preprocessed',
                        help='Path to data directory')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='Number of training epochs per trial')
    args = parser.parse_args()

    # Setup logging with subject-specific filename
    log_file = f'hpo_optuna_subject{args.subject}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Subject: {args.subject}")
    logging.info(f"Data path: {args.data_path}")

    # Load data once
    logging.info("Loading data...")
    config = Config(subject_num=args.subject)
    train_loader, valid_loader, test_loader = get_dataloaders(config,probe_map_path='../config/Precision_map_v2.xml')
    dataloaders = (train_loader, valid_loader, test_loader)
    logging.info("Data loaded successfully")

    # Create study with subject-specific name and database
    study_name = f'repeat_cnn2d_lstm_hpo_subject{args.subject}'
    db_path = f'sqlite:///hpo_optuna_subject{args.subject}.db'

    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=db_path,
        load_if_exists=True,  # Resume if exists
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    logging.info("Starting optimization...")
    logging.info(f"Study name: {study.study_name}")
    logging.info(f"Database: {db_path}")

    # Define callback to show trial results
    def callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logging.info(f"\n{'='*60}")
            logging.info(f"Trial {trial.number} finished with value: {trial.value:.4f}")
            logging.info(f"Current best value: {study.best_value:.4f} (Trial {study.best_trial.number})")
            logging.info(f"{'='*60}\n")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logging.info(f"Trial {trial.number} was pruned")

    # Run optimization
    # STAGE 1: Initial exploration - use n_repeats=1 to quickly find promising regions
    # After this completes, use validate_best_configs.py for final validation with n_repeats=5
    n_trials = args.n_trials
    n_epochs = args.n_epochs
    n_repeats = 1  # Set to 1 for fast exploration

    logging.info(f"Configuration: {n_trials} trials, {n_epochs} epochs, {n_repeats} repeat(s) per config")

    study.optimize(
        lambda trial: objective(trial,args.subject, dataloaders, device, n_epochs=n_epochs, n_repeats=n_repeats),
        n_trials=n_trials,
        timeout=None,
        n_jobs=1,  # Use 1 for GPU, increase for CPU parallelization
        show_progress_bar=True,
        callbacks=[callback]
    )

    # Print results
    logging.info("\n" + "="*50)
    logging.info("Optimization Complete!")
    logging.info("="*50)

    logging.info(f"\nBest trial: {study.best_trial.number}")
    logging.info(f"Best R2 score: {study.best_trial.value:.4f}")
    logging.info("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        logging.info(f"  {key}: {value}")

    # Save results
    results_df = study.trials_dataframe()
    results_csv = f'hpo_optuna_results_subject{args.subject}.csv'
    results_df.to_csv(results_csv, index=False)
    logging.info(f"\nResults saved to '{results_csv}'")

    # Print top 10 trials
    logging.info("\nTop 10 trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:10]
    for i, trial in enumerate(top_trials, 1):
        logging.info(f"{i}. Trial {trial.number}: R2 = {trial.value:.4f}")

    # Visualization suggestions
    logging.info("\n" + "="*50)
    logging.info("To visualize results, use:")
    logging.info(f"  optuna-dashboard {db_path}")
    logging.info(f"Or use: python plot_optimization.py --subject {args.subject}")
    logging.info("="*50)


if __name__ == "__main__":
    main()
