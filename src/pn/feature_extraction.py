import numpy as np
import os, sys, math, re
import matplotlib.pyplot as plt
import pandas as pd
import torch
import scipy.signal as signal
from pathlib import Path
import mne
from inspect import signature


from intanutil import intan_rhd as intan 
from pn.sig_proc import * 




def load_pn_ccout_from_path(file_path):
    '''
    TODO: use Path or os.path for path variables

    '''
    ###############
    # from the intan
    rhd_data = intan.read_data(file_path + '/info.rhd', dat=True, rescale_to_voltage=False)
    
    # amplifier data
    amplifier_data = rhd_data['amplifier_data']  # amplifier data 
    amplifier_timing = rhd_data['t_amplifier']
    
    ########################################
    # Arduino timing sync data from the intan 

    try:
        trigger_data = rhd_data['board_dig_in_data'][0,:]
        trigger_timing = rhd_data['t_dig']
        trigger_data = np.asarray(intan_arduino_trigger, dtype="int")
    except:
        print("No Trigger Found")
        trigger_data = None
        trigger_timing = None
    ################
    # from ephys
    
    #Load data from subfolders
    ## Joystick Vx
    try:
        with open(file_path+"continuous/TaskUIJoystickVx/continuous.dat", 'rb') as ifs:
            vx_data = np.fromfile(ifs, dtype='<i2')
        vx_data = np.asarray(vx_data, dtype='double')
        vx_timestamps = np.load(file_path+"continuous/TaskUIJoystickVx/timestamps.npy")   # # samples x1
        vx_wall_clock = np.load(file_path+"continuous/TaskUIJoystickVx/wall_clocks.npy")  
        
        
    except:
        print("No Ephys Vx Data Found")
        vx_timestamps = None
        vx_wall_clock = None
        vx_data = None
    
    #Load data from subfolders
    ## Joystick Vy
    try:
        with open(file_path+"continuous/TaskUIJoystickVy/continuous.dat", 'rb') as ifs:
            vy_data = np.fromfile(ifs, dtype='<i2')
        vy_data = np.asarray(vy_data, dtype='double')
        vy_timestamps = np.load(file_path+"continuous/TaskUIJoystickVy/timestamps.npy")   # # samples x1
        vy_wall_clock = np.load(file_path+"continuous/TaskUIJoystickVy/wall_clocks.npy")  
        
        
    except:
        print("No Ephys Vy Data Found")
        vy_timestamps = None
        vy_wall_clock = None
        vy_data = None

    # Joystick XY
    try:
        with open(file_path+"continuous/TaskUIJoystickXY/continuous.dat", 'rb') as ifs:
            xy_data = np.fromfile(ifs, dtype='<u2')
        xy_data = np.asarray(xy_data, dtype='double')
        xy_timestamps = np.load(file_path+"continuous/TaskUIJoystickXY/timestamps.npy")   # # samples x1
        xy_wall_clock = np.load(file_path+"continuous/TaskUIJoystickXY/wall_clocks.npy")  
        
        
        x_idxs = np.arange(0,len(xy_data), 2)
        y_idxs = np.arange(1,len(xy_data), 2)
        
        x_data = xy_data[x_idxs]
        y_data = xy_data[y_idxs]
        
    except:
        print("No Ephys XY Data Found")
        xy_timestamps = None
        xy_wall_clock = None
        x_data = None
        y_data = None
    ########################################
    # 
    # arduino timing sync data from the PC (Open Ephys)
    try:
        with open(file_path+"continuous/ArduinoAligmentSignal/continuous.dat", 'rb') as ifs:
            ard_data = np.fromfile(ifs, dtype='<u2')
        ard_sample_numbers = np.load(file_path+"continuous/ArduinoAligmentSignal/sample_numbers.npy")   # # samples x1
        ard_wall_clock = np.load(file_path+"continuous/ArduinoAligmentSignal/wall_clocks.npy")  
        ard_timestamps = np.load(file_path+"continuous/ArduinoAligmentSignal/timestamps.npy")   # # samples x1
    except:
        print("No Ephys Arduino Data Found")
        ard_timestamps = None
        ard_wall_clock = None
        ard_data = None
    data =   {'amplifier_data':               amplifier_data, 
              'amplifier_timing':             amplifier_timing,
              
            'trigger_data':                 trigger_data,
            'trigger_timing':               trigger_timing,
            'xy_timestamps':                xy_timestamps,
            'xy_wall_clock':                xy_wall_clock,
            'x_data':                       x_data,
            'y_data':                       y_data,
            'vx_timestamps':                vx_timestamps,
            'vx_wall_clock':                vx_wall_clock,
            'vx_data':                      vx_data,
            'vy_timestamps':                vy_timestamps,
            'vy_wall_clock':                vy_wall_clock,
            'vy_data':                      vy_data,
            'ard_data':                     ard_data,
            'ard_timestamps':               ard_timestamps,
            'ard_wall_clock':               ard_wall_clock,
            'fs':                           20000,
            'ephys_fs':                     1000000000,
           }
    try:
        if ('DiscreteCenterOutInference' in  os.listdir(os.path.join(file_path, 'continuous') ) ):
            with open(file_path+"continuous/DiscreteCenterOutInference/continuous.dat", 'rb') as ifs:
                inference_data = np.fromfile(ifs, dtype='<u2')
            inference_data = np.asarray(inference_data, dtype='double')
            x_idxs = np.arange(0,len(inference_data), 2)
            y_idxs = np.arange(1,len(inference_data), 2)
            inf_x = inference_data[x_idxs]
            inf_y = inference_data[y_idxs]
            inference_timestamps = np.load(file_path+"continuous/DiscreteCenterOutInference/timestamps.npy")   # # samples x1
            inference_wall_clock = np.load(file_path+"continuous/DiscreteCenterOutInference/wall_clocks.npy")  
        
            data['inf_x'] = inf_x
            data['inf_y'] = inf_y
            data['inf_timestamps'] = inference_timestamps
            data['inf_wall_clock'] = inference_wall_clock
    except:
        print("error trying to load inference data")
    return data

def load_bad_channel_indices_xlsx(path_xlsx, thresh_ohm=4_000_000):
    df = pd.read_csv(path_xlsx)
    imp_col = [c for c in df.columns if "impedance" in c.lower() and "ohm" in c.lower()][0]
    chan_col = [c for c in df.columns if "channel number" in c.lower()][0]
    port_col = [c for c in df.columns if c.lower().startswith("port")][0]

    def port_to_block(port_str):
        m = re.search(r"Port\s*([A-H])", str(port_str), re.I)
        return (ord(m.group(1).upper()) - ord("A")) if m else None

    def channel_in_port(ch_str):
        m = re.search(r"[A-H]-(\d+)", str(ch_str), re.I)
        return int(m.group(1)) if m else None

    blk = df[port_col].apply(port_to_block)
    ch  = df[chan_col].apply(channel_in_port)
    idx0 = blk*128 + ch
    bad  = df[imp_col] > thresh_ohm
    bad_idx = sorted(idx0[bad].dropna().astype(int).tolist())
    return bad_idx

def zero_high_imp_channels(PARAMETERS, data):
    bad_idx_list = load_bad_channel_indices_xlsx(PARAMETERS['imp_file'], PARAMETERS['imp_thresh_ohm'])
    _,T = np.shape(data['amplifier_data'])
    data['amplifier_data'][bad_idx_list] = np.zeros((len(bad_idx_list), T))

def filter_motion_data(PARAMETERS, data):
    # TODO: deal with constants
    cutoff_frequency = PARAMETERS['cutoff_frequency']  # Hz
    fs = data['fs']
    
    dx_windowed = block_moving_average(data['dx_upsampled'], 1000)
    dy_windowed = block_moving_average(data['dy_upsampled'], 1000)
    
    data['dx_filtered'] = lowpass_filter_scipy(dx_windowed, cutoff_frequency, fs)
    data['dy_filtered'] = lowpass_filter_scipy(dy_windowed, cutoff_frequency, fs)

def separate_trials_velocity(data, up_thrsh = 0.006, down_thrsh = 0.004, trial_time_s = 4, base_time_s = 0.4):
    #TODO: deal with constants
    
    fs = data["fs"]
    
    filt_zero = np.abs(data['dx_filtered'])+np.abs(data['dy_filtered'])
    
    trial_state = "rest"

    trial_time = int(trial_time_s*fs)
    baseline_time = int(base_time_s*fs)
    
    start_idxs = []
    end_idxs = []
    
    t = 0
    while (t < len(filt_zero)):
        match trial_state:
            case "rest":
                if (filt_zero[t] > up_thrsh):
                    start_idxs.append(t-baseline_time)
                    trial_state = "moving"
                    t += 100
                else:
                    t += 1
            case "moving":
                if(filt_zero[t] < down_thrsh):
                    end_idxs.append(t)
                    trial_state = "rest"
                    t += 100
                else:
                    t+=1
            case _:
                print("invalid state")
    start_idxs = np.asarray(start_idxs, dtype = "int")
    end_idxs = np.asarray(end_idxs)
    
    #start_idxs = start_idxs[4:]
    #end_idxs = end_idxs[4:]
    
    too_long = np.where((end_idxs-start_idxs) > trial_time) 
    
    start_idxs = np.delete(start_idxs, too_long)
    end_idxs = np.delete(end_idxs, too_long)
    
    too_short = np.where((end_idxs-start_idxs)/fs < 2)
    start_idxs = np.delete(start_idxs, too_short)
    end_idxs = np.delete(end_idxs, too_short)
    
    print(f'Discarded {len(too_long)} tasks for being too long')
    print(f'Discarded {len(too_short)} tasks for being short')
    start_idxs = start_idxs - baseline_time

    ## really this can be combined with the previous one 
    trial_lens = []
    
    trial_list = []
    for i in range(len(start_idxs)):
        s = start_idxs[i]
        e = end_idxs[i]
        num_electrodes, _ = np.shape( data['amplifier_data_aligned']  )
        zero_padding = int(trial_time + baseline_time - (e-s) )
    
        padded_amp_data = data['amplifier_data_aligned'][:,s:s+int(4*fs)]
    
        trial_list.append( Trial( np.transpose(data['amplifier_data_aligned'][:,s:s+int(baseline_time-0.1)]),  # baseline ecog -- for now we're just taking one for all trials in file
                                 np.hstack([ data['x_upsampled'][s:e], np.nan*np.ones(zero_padding)])  , # x pos
                                 np.hstack([ data['y_upsampled'][s:e], np.nan*np.ones(zero_padding)]), # y pos
                                 np.hstack([ data['vx_upsampled'][s:e], np.nan*np.ones(zero_padding)]),  # dx
                                 np.hstack([ data['vy_upsampled'][s:e], np.nan*np.ones(zero_padding)]),
                                 np.transpose(padded_amp_data), 
                                 0,       # trial_type, I don't think this gets used either
                                 fs 
                                )
                            )
        trial_list[-1].set_id(i)
        trial_lens.append(e-s)
    return trial_list

class Trial():
    def __init__(self,becog, ax, ay, adx, ady, aecog, trial_type,fs):
        self.active_x = ax
        self.active_y = ay
        self.active_dx = adx
        self.active_dy = ady
        self.active_ecog = aecog

        self.baseline_ecog = becog
        self.trial_type = trial_type
        self.features = {}
        self.labels = {}
        self.fs = fs
        self.filt_data = None
        self.trial_id = None
    def add_features(self, features, feature_label):
        self.features[feature_label] = features
    def add_labels(self, labels, label_name):
        self.labels[label_name] = labels
    def set_filt_data(self,data):
        self.filt_data = data
    def set_id(self, id):
        self.trial_id = id

def extract_ssep_basic(PARAMETERS, trial):
    trial.active_ecog = butter_bandpass_filter(trial.active_ecog, [4,300], fs=20000, order=5)
    trial.baseline_ecog = butter_bandpass_filter(trial.active_ecog, [4,300], fs=20000, order=5)
    
    data_window = np.array((0,100)).astype(int)
    data_window *= int(Fs_raw/1000)
    amp_trial = np.zeros((200, 1024, int((data_window[1]-data_window[0]))))
    count = 0 # Using count instead of enumerate here, in case you want to add critera for rejecting individual trials 
    for t in tqdm(trig[:200]):
        amp_trial[count, selected_chs, :] = neural_bp[selected_chs,t+data_window[0]:t+data_window[1]]
        count += 1
    ssep = np.mean(amp_trial,axis=0)
    trial.add_features(ssep, 'ssep')