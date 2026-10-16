import numpy as np
import os,json, math


from datetime import datetime

from typing import List, Tuple, Optional
import file_loader



from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import p5_utils as p5u

def     electrode_map(select_channels: Optional[List[str]] = None,
                 probe_map_path: str = './config/Precision_map_v2.xml') -> Tuple[List[int], List[int]]:
    """
    Maps electrode channel names to their physical coordinates on the probe.
    
    This function takes a list of channel names and returns their corresponding
    x and y coordinates based on the probe map configuration file.
    
    Args:
        select_channels: List of channel names to map to coordinates.
                        If None, defaults to all inbound electrodes from config.
        probe_map_path: Path to the XML file containing the probe layout mapping.
                        Defaults to the Precision probe map version 2.
    
    Returns:
        Tuple containing two lists:
        - First list: x-coordinates of the selected electrodes
        - Second list: y-coordinates of the selected electrodes
        
    Raises:
        ValueError: If any channel name is not found in the probe map.
        
    Example:
        >>> x_coords, y_coords = electrode_map(['chan1', 'chan5', 'chan10'])
        >>> # Returns coordinates for the specified channels
    """
    probe_map, _ = file_loader.parse_probe_map(probe_map_path, n_channels=1024)
    electrode_x = []
    electrode_y = []
    if not select_channels:
        electrode_file = os.path.join( os.path.dirname(probe_map_path), 'electrode_inbound.npy')
        select_channels = np.load(electrode_file)
        print(f"Using default channels, all inbound electrodes will be included.")
    for chan in select_channels:
        if chan in probe_map.keys():
            electrode_x.append(probe_map[chan][0])
            electrode_y.append(probe_map[chan][1])
        else:
            raise ValueError(f"Channel {chan} not found in probe map.")
    return electrode_x, electrode_y

#--------------------------------------
def pos_to_electrode_map(probe_map_path):

    probe_map, probe_map_ordered = file_loader.parse_probe_map(probe_map_path, n_channels=1024)

    electrode_x, electrode_y = electrode_map(probe_map_path = probe_map_path)

    max_dims = [np.max(electrode_x)+1,np.max(electrode_y)+1]
    num_electrodes = len(electrode_x)

    pos_to_electrode = np.zeros(max_dims)
    for e in range(num_electrodes):
        e_x = electrode_x[e]
        e_y = electrode_y[e]
        pos_to_electrode[e_x][e_y] = e+1
    return pos_to_electrode




def get_electrode_subsets(pos_to_electrode, sizes, density):
    electrode_subsets = []
    max_dims = np.shape(pos_to_electrode)
    spacing = int(1/density)
    for size in sizes:
        for x in range(max_dims[0]-size*spacing):
            for y in range(max_dims[1]-size*spacing):
                x_set = [i for i in range(x, x+size*spacing, spacing)]
                y_set =  [i for i in range(y, y+size*spacing, spacing)]
                potential_subset = pos_to_electrode[x_set,:]
                potential_subset = potential_subset[:,y_set]
                potential_subset = potential_subset.flatten()
                if (0  not in potential_subset):
                    electrode_subsets.append((x_set,y_set))
    return electrode_subsets


def get_electrode_subsets_density(pos_to_electrode, max_spacing):
        electrode_subsets = []
        max_dims = np.shape(pos_to_electrode)
        print(f'max dims {max_dims}')
        for space in range(1,max_spacing+1):
            x_set = [i for i in range(0, max_dims[0]-1, space)]
            y_set = [i for i in range(0, max_dims[1]-1, space)]

            potential_subset = pos_to_electrode[x_set,:]
            potential_subset = potential_subset[:,y_set]
            potential_subset = potential_subset.flatten()

            electrode_subsets.append((x_set,y_set))
        return electrode_subsets
def plot_velocity_predictions(preds, tgts, score, r2x, r2y, figsize=(15, 7), linewidth=0.8, 
                            save_path=None, dpi=300, bbox_inches='tight'):
    """
    Plot velocity predictions vs ground truth for vx and vy components.
    
    Parameters:
    -----------
    preds : array-like, shape (n_samples, 2)
        Predicted velocities [vx, vy]
    tgts : array-like, shape (n_samples, 2) 
        Ground truth velocities [vx, vy]
    score : float
        Overall R² score for combined vx & vy
    r2x : float
        R² score for vx component
    r2y : float
        R² score for vy component
    figsize : tuple, optional
        Figure size (width, height)
    linewidth : float, optional
        Line width for plots
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, optional
        DPI for saved figure
    bbox_inches : str, optional
        Bbox setting for saved figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    axs : array of matplotlib.axes.Axes
        Array of subplot axes
    """
    import matplotlib.pyplot as plt
    
    print(f"R² (vx & vy, raw), (vx, raw), (vy, raw) = {score:.3f}, {r2x:.3f}, {r2y:.3f}")

    fig, axs = plt.subplots(2, 1, figsize=figsize)

    axs[0].plot(preds[:, 0], label='raw inference vx', linewidth=linewidth)
    axs[0].plot(tgts[:, 0], label='gr vx', linewidth=linewidth)
    axs[0].set_title(f"r2x {r2x}")

    axs[1].plot(preds[:, 1], label='raw inference vy', linewidth=linewidth)
    axs[1].plot(tgts[:, 1], label='gr vy', linewidth=linewidth)
    axs[1].set_title(f"r2y {r2y}")

    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    
    # Save figure if path provided
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Figure saved to: {save_path}")
    
    return fig, axs

def save_experiment_config(model_configs, config_file, experiment_params, 
                          experiment_version, subject_list, data_root):
    """Save the model configurations and experiment parameters to a JSON file"""
    
    # Convert model classes to string names for JSON serialization
    serializable_configs = {}
    for model_name, config in model_configs.items():
        serializable_configs[model_name] = {
            'model_class_name': config['model_class'].__name__,
            'config_updates': config['config_updates']
        }
    
    experiment_data = {
        'experiment_info': {
            'version': experiment_version,
            'timestamp': datetime.now().isoformat(),
            'description': f'Model architecture comparison experiment v{experiment_version}'
        },
        'experiment_params': experiment_params,
        'model_configs': serializable_configs,
        'subjects_tested': subject_list,
        'data_root': data_root
    }
    
    with open(config_file, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"Experiment configuration saved to: {config_file}")
    return experiment_data

def load_experiment_config(config_file):
    """Load experiment configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)
    
def compare_experiment_configs(config_file1, config_file2):
    """Compare two experiment configurations and highlight differences"""
    
    config1 = load_experiment_config(config_file1)
    config2 = load_experiment_config(config_file2)
    
    print(f"\nComparing experiments:")
    print(f"Experiment 1 (v{config1['experiment_info']['version']}): {config_file1}")
    print(f"Experiment 2 (v{config2['experiment_info']['version']}): {config_file2}")
    print("-" * 50)
    
    # Compare experiment parameters
    params1 = config1['experiment_params']
    params2 = config2['experiment_params']
    
    print("Parameter differences:")
    for key in set(params1.keys()) | set(params2.keys()):
        val1 = params1.get(key, "NOT SET")
        val2 = params2.get(key, "NOT SET")
        if val1 != val2:
            print(f"  {key}: {val1} -> {val2}")
    
    # Compare model configurations
    models1 = set(config1['model_configs'].keys())
    models2 = set(config2['model_configs'].keys())
    
    print(f"\nModels tested:")
    print(f"  Only in v{config1['experiment_info']['version']}: {models1 - models2}")
    print(f"  Only in v{config2['experiment_info']['version']}: {models2 - models1}")
    print(f"  Common models: {models1 & models2}")
    
    return config1, config2

def compare_model_results(csv_file1, csv_file2):
    """Compare results between two different experiment runs"""
    import pandas as pd
    
    try:
        df1 = pd.read_csv(csv_file1)
        df2 = pd.read_csv(csv_file2)
        
        # Remove error rows and convert to numeric
        for df in [df1, df2]:
            df = df[df['R2'] != 'ERROR']
            df['R2'] = pd.to_numeric(df['R2'])
        
        # Get experiment versions
        v1 = df1['ExperimentVersion'].iloc[0] if 'ExperimentVersion' in df1.columns else "1"
        v2 = df2['ExperimentVersion'].iloc[0] if 'ExperimentVersion' in df2.columns else "2"
        
        print(f"\nComparing Model Results: v{v1} vs v{v2}")
        print("="*50)
        
        # Get mean R2 for each model in both experiments
        mean1 = df1.groupby('ModelName')['R2'].mean()
        mean2 = df2.groupby('ModelName')['R2'].mean()
        
        # Find common models
        common_models = set(mean1.index) & set(mean2.index)
        
        if common_models:
            print("Performance comparison for common models:")
            for model in sorted(common_models):
                diff = mean2[model] - mean1[model]
                direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
                print(f"  {model}: {mean1[model]:.4f} -> {mean2[model]:.4f} ({direction} {abs(diff):.4f})")
        
        # Models only in one experiment
        only_in_1 = set(mean1.index) - set(mean2.index)
        only_in_2 = set(mean2.index) - set(mean1.index)
        
        if only_in_1:
            print(f"\nModels only in v{v1}: {list(only_in_1)}")
        if only_in_2:
            print(f"Models only in v{v2}: {list(only_in_2)}")
            
    except Exception as e:
        print(f"Error comparing results: {e}")

def gaussian_smooth_time(x: torch.Tensor, fs_hz: float, sigma_ms: float = 40.0,
                         *, truncate: float = 3.0, fwhm: bool = False,
                         pad_mode: str = "reflect") -> torch.Tensor:
    if fwhm:
        sigma_ms = sigma_ms / 2.355

    sigma_samples = (sigma_ms / 1000.0) * fs_hz
    sigma_samples = max(float(sigma_samples), 1e-6)
    half_width = int(math.ceil(truncate * sigma_samples))
    klen = 2 * half_width + 1

    device, dtype = x.device, x.dtype
    t = torch.arange(-half_width, half_width + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(t ** 2) / (2 * (sigma_samples ** 2)))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, klen)

    orig_shape = x.shape
    if x.ndim == 3:
        H, W, T = x.shape
        x2 = x.reshape(H * W, T).unsqueeze(0)  # (1,C,T)
        C = H * W
    elif x.ndim == 2:
        C, T = x.shape
        x2 = x.unsqueeze(0)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {x.shape}")

    xpad = F.pad(x2, (half_width, half_width), mode=pad_mode)
    y = F.conv1d(xpad, kernel.expand(C, 1, -1), groups=C)  # (1,C,T)

    y = y.squeeze(0)
    if len(orig_shape) == 3:
        y = y.reshape(orig_shape)
    return y







def CNN3D_test(model, test_loader, config):
    model.eval()
    test_loss = 0.0
    test_count = 0
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to('cuda')
            y = y.to('cuda')
            x = x.unsqueeze(-1) # adding a feature dim
            y = y.permute(0,2,1)
            vel_targets = y[..., 0:2]
            vel_seq = model(x)
            tgt_last = vel_targets[:, -1, :]
    
            
            all_preds.append(vel_seq.cpu().numpy())
            all_tgts.append(tgt_last.cpu().numpy())
            test_count += x.size(0)
            
        test_loss /= test_count
        preds = np.concatenate(all_preds, axis=0)
        tgts = np.concatenate(all_tgts, axis=0)
        r2x = r2_score(tgts[:, 0], preds[:, 0])
        r2y = r2_score(tgts[:, 1], preds[:, 1])
        score = r2x + r2y
    return score


def CNN3D_predict(model, test_loader):
    model.eval()
    test_loss = 0.0
    test_count = 0
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to('cuda')
            y = y.to('cuda')
            x = x.unsqueeze(-1) # adding a feature dim
            y = y.permute(0,2,1)
            vel_targets = y[..., 0:2]
            vel_seq = model(x)
            tgt_last = vel_targets[:, -1, :]

            
            all_preds.append(vel_seq.cpu().numpy())
            all_tgts.append(tgt_last.cpu().numpy())
            test_count += x.size(0)
            
        test_loss /= test_count
        preds = np.concatenate(all_preds, axis=0)
        tgts = np.concatenate(all_tgts, axis=0)
        r2x = r2_score(tgts[:, 0], preds[:, 0])
        r2y = r2_score(tgts[:, 1], preds[:, 1])
        score = r2x + r2y

    return tgts, preds


def split_preprocessed_pn(tgts, preds, thrsh = 0.15, plot_thrsh=True):
    dx = tgts[:,0]
    dy = tgts[:,1]

    mvmt = dx**2 + dy**2 
    T = len(mvmt)
    if(plot_thrsh):
        fig, ax = plt.subplots(1,1, figsize = (15,5));
        plt.plot(mvmt)
        plt.plot([0,T], [thrsh, thrsh])

    
    ## Split into trials 
    start_idxs = []
    end_idxs = []

    T = len(mvmt)
    mstate = "rest"
    for t in range(T):
        match mstate:
            case "rest":
                if mvmt[t] > thrsh:
                    start_idxs.append(t)
                    mstate = "moving"
            case "moving":
                if mvmt[t] < thrsh:
                    end_idxs.append(t)
                    mstate = "rest"
            case _:
                print("invalid state")
    x_values = []
    y_values = []
    angle_bins = []
    x_hat_values = []
    y_hat_values = []

    

    for trial in range(len(start_idxs)):
        t1 = start_idxs[trial]
        t2 = end_idxs[trial]
    
        dx = tgts[t1 : t2,0],#actove dx
        dx = np.squeeze(dx) 
    
        dy = tgts[t1:t2,1], #active dy
        dy = np.squeeze(dy)
        
        dx_hat = preds[t1 : t2,0]
        dx_hat = np.squeeze(dx_hat)
        dy_hat = preds[t1 : t2,1]
        dy_hat = np.squeeze(dy_hat)

        x_ =  np.cumsum(dx)  
        y_ =  np.cumsum(dy)  
        xhat_ =  np.cumsum(dx_hat)  
        yhat_ =  np.cumsum(dy_hat) 
    
        angle = 180/np.pi* np.atan2(  y_[-1] - y_[0], x_[-1] - x_[0] ) 
    
        angle_class = int( np.rint(angle / 45) + 4)
        if angle_class == 8:
            angle_class = 0

        x_values.append(x_)
        y_values.append(y_)    
        x_hat_values.append(xhat_)
        y_hat_values.append(yhat_)
        angle_bins.append(angle_class)

    return {'x_values': x_values,
            'y_values': y_values,
            'x_hat_values': x_hat_values,
            'y_hat_values': y_hat_values,
            'angle_bins': angle_bins, 
            'num_trials': len(angle_bins)
            }
        





