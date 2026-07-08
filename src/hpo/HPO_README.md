# Hyperparameter Optimization for CNN2D_LSTM

This directory contains scripts for optimizing the hyperparameters of the CNN2D_LSTM model using Optuna.

## Overview

The HPO system uses **Bayesian optimization** via Optuna to efficiently search the hyperparameter space, which is much more efficient than grid search. Key features:

- **Automated hyperparameter tuning** for all major model parameters
- **Pruning** of unpromising trials to save compute time
- **Persistence** via SQLite database - can resume interrupted runs
- **Visualization** tools for analyzing results
- **Comparison** with previous grid search results

## Installation

Install the required packages:

```bash
pip install optuna optuna-dashboard pandas matplotlib seaborn scikit-learn torch
```

## Quick Start (Two-Stage Approach)

**Important:** HPO is conducted on a **per-participant basis**. Each subject (2, 3, or 4) has separate databases and results.

### Stage 1: Initial Exploration

Run fast exploration with single runs per configuration for a specific participant:

```bash
cd hpo
python hpo_optuna.py --subject 3
```

**Command-line options:**
- `--subject`: Subject number (2, 3, or 4). Default: 3
- `--n-trials`: Number of trials (default: 100)
- `--n-epochs`: Epochs per trial (default: 100)
- `--data-path`: Path to data directory

This will:
- Run 100 trials with `n_repeats=1` (fast exploration)
- Test diverse hyperparameter combinations
- Save results to `hpo_optuna_subject3.db` and `hpo_optuna_results_subject3.csv`
- Log progress to `hpo_optuna_subject3.log` and console

**Note:** You can stop and resume the optimization at any time. The database saves all progress.

### Stage 2: Validate Best Configurations

After Stage 1 completes, validate the top configurations with multiple runs:

```bash
python validate_best_configs.py --subject 3 --top-n 10 --n-repeats 5
```

**Command-line options:**
- `--subject`: Subject number (2, 3, or 4). Default: 3
- `--top-n`: Number of top configs to validate (default: 10)
- `--n-repeats`: Runs per config (default: 5)
- `--n-epochs`: Training epochs (default: 100)
- `--data-path`: Path to data directory

This will:
- Load the top 10 configurations from Stage 1 for the specified subject
- Run each configuration 5 times to assess variance
- Report median, mean ± std for each
- Save the best configuration to `best_config_subject3_[timestamp].json`

### Running for Multiple Participants

To optimize for all participants, run the stages separately for each:

```bash
# Subject 2
python hpo_optuna.py --subject 2
python validate_best_configs.py --subject 2

# Subject 3
python hpo_optuna.py --subject 3
python validate_best_configs.py --subject 3

# Subject 4
python hpo_optuna.py --subject 4
python validate_best_configs.py --subject 4
```

Each participant will have their own database, logs, and results files.

### Visualize Results

```bash
python plot_optimization.py --subject 3
```

This generates:
- `plots/optimization_history.png` - Performance over trials
- `plots/param_importances.png` - Which parameters matter most
- `plots/parallel_coordinate.png` - Parameter combinations
- `plots/slice_plot.png` - Individual parameter effects
- `plots/contour_plot.png` - Parameter interactions
- `plots/custom_analysis.png` - Detailed analysis dashboard
- `plots/grid_vs_optuna.png` - Comparison with grid search

### Interactive Dashboard (Optional)

For an interactive web dashboard for a specific subject:

```bash
optuna-dashboard sqlite:///hpo_optuna_subject3.db
```

Then open your browser to the displayed URL (usually http://localhost:8080).

## Hyperparameters Being Optimized

### CNN Parameters
- `num_kernels`: Number of 3D convolution kernels (2-8)
- `spatial_kernel_size`: Spatial filter size (4-10)
- `temporal_kernel_size`: Temporal filter size (20-60)
- `spatial_kernel_stride`: Spatial stride (1-3)
- `temporal_kernel_stride`: Temporal stride (3-10)

### RNN Parameters
- `rnn_hidden_size`: LSTM hidden units (128, 256, 384, 512, 640, 800)
- `rnn_layers`: Number of LSTM layers (2-6)

### Regularization
- `cnn_dropout`: CNN dropout rate (0.2-0.7)
- `rnn_dropout`: RNN dropout rate (0.1-0.5)

### Training
- `learning_rate`: Adam learning rate (1e-5 to 1e-3, log scale)
- `weight_decay`: L2 regularization (1e-5 to 1e-2, log scale)

## Configuration

### Modify Search Space

Edit the `objective()` function in [hpo_optuna.py](hpo_optuna.py:299):

```python
# Example: Change RNN hidden size options
config.rnn_hidden_size = trial.suggest_categorical(
    'rnn_hidden_size',
    [128, 256, 512, 1024]  # Your custom values
)

# Example: Change learning rate range
config.learning_rate = trial.suggest_float(
    'learning_rate',
    1e-6,  # min
    1e-2,  # max
    log=True
)
```

### Modify Number of Trials

Edit `main()` in [hpo_optuna.py](hpo_optuna.py:428):

```python
n_trials = 100  # Change this number
```

### Modify Training Epochs

Edit `main()` in [hpo_optuna.py](hpo_optuna.py:429):

```python
n_epochs = 100  # Change this number
```

### Run Each Configuration Multiple Times

To account for training variance (like your grid search), edit `main()` in [hpo_optuna.py](hpo_optuna.py:430):

```python
n_repeats = 3  # Run each config 3 times and return the median score
```

**Note:** This significantly increases training time. For 100 trials with 3 repeats, you'll train 300 models.

**How it works:** When `n_repeats > 1`, each configuration is trained multiple times with different random initializations. The **median** score is returned (more robust to outliers than mean), along with mean ± std for reference.

**Recommendation:** Start with `n_repeats=1` to quickly explore the space, then use `n_repeats=3` or `n_repeats=5` for final validation of the best configurations.

## Understanding Results

### Best Hyperparameters

Check the console output or `hpo_optuna.log` for:
```
Best trial: 42
Best R2 score: 1.6234
Best hyperparameters:
  num_kernels: 4
  rnn_hidden_size: 512
  ...
```

### Results CSV

`hpo_optuna_results.csv` contains all trials with:
- Trial number
- Parameter values
- R2 score
- Trial state (COMPLETE, PRUNED, FAIL)

### Parameter Importance

The `param_importances.png` plot shows which parameters have the biggest impact on performance. Focus on tuning these first.

## Advanced Usage

### Resume Optimization

Simply run `python hpo_optuna.py` again. It will automatically resume from the database.

### Multi-Objective Optimization

To optimize for multiple metrics (e.g., R2 score and model size):

```python
def objective(trial, dataloaders, device):
    # ... train model ...

    # Calculate metrics
    r2_score = evaluate_model(model, test_loader, device)
    model_size = sum(p.numel() for p in model.parameters())

    return r2_score, -model_size  # Maximize R2, minimize size

# In main():
study = optuna.create_study(
    directions=['maximize', 'minimize']  # Multi-objective
)
```

### Parallel Optimization (CPU only)

For CPU training, you can run multiple trials in parallel:

```python
study.optimize(
    lambda trial: objective(trial, dataloaders, device),
    n_trials=100,
    n_jobs=4  # 4 parallel workers
)
```

**Note:** For GPU training, keep `n_jobs=1` to avoid OOM errors.

### Custom Pruning Strategy

Change the pruner in `main()`:

```python
# Median Pruner (default)
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

# Hyperband Pruner (aggressive)
pruner = optuna.pruners.HyperbandPruner()

# No pruning
pruner = optuna.pruners.NopPruner()

study = optuna.create_study(pruner=pruner, ...)
```

## Tips for Better Results

1. **Start with fewer epochs** (e.g., 20) to quickly explore the space, then run longer trials with best configs
2. **Use pruning** to stop bad trials early and save compute
3. **Check parameter importances** to focus tuning on what matters
4. **Run multiple short studies** rather than one very long one
5. **Visualize results frequently** to understand what's working

## Comparison with Grid Search

Your previous grid search from `tuning.py`:
- Tested specific parameter combinations manually
- Required running all combinations to completion
- No intelligent search strategy

Optuna HPO:
- Intelligently samples promising regions
- Prunes unpromising trials early
- Typically finds better results with fewer trials
- Can be easily resumed and extended

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in `get_dataloaders()`
- Reduce `n_epochs` in `main()`
- Use smaller parameter ranges

### Slow Training
- Enable pruning more aggressively
- Reduce `n_warmup_steps` in pruner
- Start with fewer epochs

### Database Locked
- Only run one `hpo_optuna.py` instance at a time
- Delete `hpo_optuna.db` to start fresh (loses all history)

## Files

- **hpo_optuna.py** - Main optimization script
- **plot_optimization.py** - Visualization script
- **hpo_optuna.db** - SQLite database with all results
- **hpo_optuna_results.csv** - Trial results in CSV format
- **hpo_optuna.log** - Execution logs
- **plots/** - Generated visualization plots

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna Examples](https://github.com/optuna/optuna-examples)
- [Hyperparameter Optimization Guide](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
