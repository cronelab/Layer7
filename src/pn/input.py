import numpy as np
import os,json, math

import torch
import torch.nn.functional as F

from datetime import datetime

from typing import List, Tuple, Optional
from pn import file_loader

def electrode_map(select_channels: Optional[List[str]] = None,
                 probe_map_path: str = '../../config/Precision_map_v2.xml') -> Tuple[List[int], List[int]]:
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








