# ----------------------------------------------------------
# Create by: Elton Ho
# Create date: 10-NOV-2022
# Last updated: 21-DEC-2022
#
# All things directly I/O with files
# ----------------------------------------------------------

import xml.etree.ElementTree as ET
import numpy as np
import csv
import os
import sys
import logging
import joblib, pickle
import yaml
import time


def parse_probe_map(file, n_channels=512):
    """Parse probe map xml file into electrode coordinates
    Usage: probe_map, probe_map_ordered = parse_probe_map(filepath, n_channels=512)

    Parameters:
    file : file path to target xml file, including "xxx.xml", or file object
    n_channels : number of channels to be parsed. Currently only supporting 512 and 1024.

    Returns:
    probe_map : dict : Look-up table for channel name to coordinates, e.g. 'A-001' : (140,15)
    probe_map_ordered : numpy array : 2D Array of coordinates ordered from A-000 to H-127

    Last updated: 21-DEC-2022
    DRI: Elton Ho
    """
    # probe_map_tree = ET.parse(file)
    # probe_map_root = probe_map_tree.getroot()
    # probe_map = {}
    # probe_map_ordered = np.zeros((n_channels, 2))
    # n_ports = int(n_channels / 128)
    # available_ports = ["A", "B", "C", "D", "E", "F", "G", "H"][:n_ports]

    # CH_LINE_SKIPS = {
    #     512: 9,
    #     1024: 13,
    # }
    # try:
    #     line_skip = CH_LINE_SKIPS[n_channels]
    # except KeyError as err:
    #     err_msg = f"n_channels must be one of {list(CH_LINE_SKIPS.keys())}."
    #     logging.error(err_msg)
    #     logging.exception(err)
    #     raise ValueError(err_msg)

    # for port in np.arange(line_skip, line_skip + n_ports):
    #     for ch in np.arange(128):
    #         try:

    #             ch_data = probe_map_root[0][0][port][ch]
    #             ch_name = (
    #                 available_ports[port - line_skip]
    #                 + "-"
    #                 + ch_data.attrib["channelNumber"].zfill(3)
    #             )

    #             probe_map[ch_name] = (
    #                 int(ch_data.attrib["x"]),
    #                 int(ch_data.attrib["y"]),
    #             )
    #             probe_map_ordered[(port - line_skip) * 128 + ch] = (
    #                 int(ch_data.attrib["x"]),
    #                 int(ch_data.attrib["y"]),
    #             )
    #         except:
    #             probe_map_ordered[(port - line_skip) * 128 + ch] = (None, None)

    # return probe_map, probe_map_ordered


def parse_probe_map(file, n_channels=512):
    """Parse probe map xml file into electrode coordinates
    Usage: probe_map, probe_map_ordered = parse_probe_map(filepath, n_channels=512)

    Parameters:
    file : file path to target xml file, including "xxx.xml", or file object
    n_channels : number of channels to be parsed. Currently only supporting 512 and 1024.

    Returns:
    probe_map : dict : Look-up table for channel name to coordinates, e.g. 'A-001' : (140,15)
    probe_map_ordered : numpy array : 2D Array of coordinates ordered from A-000 to H-127

    Last updated: 21-DEC-2022
    DRI: Elton Ho
    """
    probe_map_tree = ET.parse(file)
    probe_map_root = probe_map_tree.getroot()
    probe_map = {}
    probe_map_ordered = np.full((n_channels, 2),-999)
    n_ports = int(n_channels / 128)
    available_ports = ["A", "B", "C", "D", "E", "F", "G", "H"][:n_ports]

    CH_LINE_SKIPS = {
        512: 9,
        1024: 13,
    }
    try:
        line_skip = CH_LINE_SKIPS[n_channels]
    except KeyError as err:
        err_msg = f"n_channels must be one of {list(CH_LINE_SKIPS.keys())}."
        logging.error(err_msg)
        logging.exception(err)
        raise ValueError(err_msg)

    for port in np.arange(line_skip, line_skip + n_ports):
        for ch in np.arange(128):
            
            try:
                ch_data = probe_map_root[0][0][port][ch]
                ch_name = (
                    available_ports[port - line_skip]
                    + "-"
                    + ch_data.attrib["channelNumber"].zfill(3)
                )

                probe_map[ch_name] = (
                    int(ch_data.attrib["x"]),
                    int(ch_data.attrib["y"]),
                )
                probe_map_ordered[(port - line_skip) * 128 + int(ch_data.attrib["channelNumber"])] = (
                    int(ch_data.attrib["x"]),
                    int(ch_data.attrib["y"]),
                )
            except:
                continue
                # print(f'skipped {(port - line_skip) * 128 + int(ch_data.attrib["channelNumber"])}')
                # probe_map_ordered[(port - line_skip) * 128 + int(ch_data.attrib["channelNumber"])] = (None, None)

    return probe_map, probe_map_ordered

def load_trial_data(file_trial_data):
        
    if isinstance(file_trial_data, str) and file_trial_data.endswith('.pkl'):
        # path_source = paths["pickle_converted_from_dat"]
        print('Loading already converted pickle file: ' + file_trial_data)    
        with open(file_trial_data, 'rb') as f:
            data_trial = pickle.load(f)
            
    elif isinstance(file_trial_data, str) and file_trial_data.endswith('.joblib'):
        # path_source = paths["pickle_converted_from_dat"]
        print('Loading already converted joblib file: ' + file_trial_data)    
        # with open(path_source, 'rb') as f:
        data_trial = joblib.load(file_trial_data)

    return data_trial

# ----------------------------------------------------------

def import_params(params_filepath):
    """Import parameters from params.yaml to populate params for general ML work
    Usage: params = import_params(params_filepath)

    Parameters:
    params_filepath : str :  parameters yaml file path

    Returns:
    params : dict : parameters related to everything ML
    See documentation (TBD) for list of parameters

    Last updated: 3-NOV-2023
    DRI: Elton Ho
    """

    with open(params_filepath, "rb") as params_file:
        params = yaml.load(params_file, Loader=yaml.Loader)

    # params = yaml.load(params_handle, Loader=yaml.Loader)
    print(params)
    return params

# ----------------------------------------------------------

def get_rhd_folders(data_folder):
    """Get list of all experiment subfolders in target folder
    Usage: exp_list = get_rhd_folders(data_folder)

    Parameters:
    data_folder: str : file path to folder containing experiment folders. Do not input folder that contains rhd files

    Returns:
    exp_list :  list ( str ) : subfolders in data folder, each representing one experiment

    Last updated: 04-NOV-2023
    DRI: Elton Ho
    Contributor: David Williams
    """

    dirs = []
    for dir_path, dir_names, file_names in os.walk(data_folder):
        if any([name.endswith(".rhd") for name in file_names]):
            dirs.append(dir_path)

    return dirs

# ----------------------------------------------------------

def load_labels(label_files=None, params=None):
    
    timer = time.time()
    labels = []

    if label_files == None:
        label_files = params['paths']['label_files']

    for df in label_files:
        file_root = df.split('\\')[-1]
        with open(df, 'rb') as flabel:
            lines = flabel.readlines()
            for l in lines:
                labels.append(int(l))

    print(f"Total parsed labels: {len(labels)}")
    print(f"Elapsed time: {time.time() - timer}")

    return labels