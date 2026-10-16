
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import r2_score

from pn.input import *
import pn.p5_utils as p5u
class Config:
    def __init__(self, subject_num, use_general_params = False,main_data_path = 'D:\\Becca\\Data\\PN_preprocessed'):
        # experiment parameters 


        # Subject Info
        #self.main_data_path = '/home/becca/Data/PN_preprocessed/'  # on corticom training
        self.main_data_path = main_data_path # on lab windows box
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        self.probe_map_path = PROJECT_ROOT.joinpath("config", "Precision_map_v2.xml")

        self.patience = 5
        #self.smooth_train_ms = None
        #self.smooth_val_ms = None
        #self.fs_hz = None

        self.smooth_train_ms = 40.0
        self.smooth_val_ms = 100.0
        self.fs_hz = 100
        
        match subject_num:
            case 3:
                self.patient = Path(self.main_data_path + "/NSR-004-003").expanduser()
                self.recordings = [
                    self.patient.joinpath("Training block 1_FIH0050blue__loc4_6_1744227575794_250409_153935_hga"),
                    self.patient.joinpath("Training block 2_FIH0050blue__loc8_2_1744227839734_250409_154359_hga"),
                    self.patient.joinpath("Training block 3_FIH0050blue__loc4_2_1744228097783_250409_154817_hga"),
                    self.patient.joinpath("Validation block_FIH0050blue_loc8_3_1744228487511_250409_155447_hga"),  # validation
                    self.patient.joinpath("Validation block_FIH0050blue_loc8_3_1744228487511_250409_155447_hga"), # test
                ]
              
                self.trial_length = 400    
                self.num_kernels = 6
                self.spatial_kernel_size = 4
                self.temporal_kernel_size = 20 
                self.spatial_kernel_stride = 3
                self.temporal_kernel_stride = 7
                self.spatial_kernel_padding =  0 #int(self.spatial_kernel_stride/2)
                self.temporal_kernel_padding =  0 #int(self.temporal_kernel_stride/2)

                # calculat post CNN shape
                self.xy_dim = int(( 33 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
                t_dim = int(( self.trial_length - self.temporal_kernel_size) / self.temporal_kernel_stride )  + 1
                self.in_channels: int = self.xy_dim**2 *self.num_kernels

                self.rnn_hidden_size: int = 384
                self.n_features: int = 1
                self.n_labels: int = 2
                self.bidirectional: bool = True
                self.batch_size: int = 64
                self.rnn_layers: int = 2 

                self.cnn_dropout: float = 0.3 
                self.rnn_dropout: float = 0.35

                # Training
                self.epochs: int = 30
                self.weight_decay: float = 6.5e-4 #2e-3
                self.learning_rate: float = 3e-4 #1e-4

            case 2:
                self.patient = Path(self.main_data_path + "/NSR-004-002").expanduser()
            
                self.recordings = [
                    self.patient.joinpath("Training block 1_green__loc4_7_1742409770184_250319_144250_hga"),
                    self.patient.joinpath("Training block 2_green__loc8_1_1742410047212_250319_144727_hga"),
                    self.patient.joinpath("Training block 3_green__loc4_3_1742410337516_250319_145217_hga"),
                    self.patient.joinpath("Validation block_green_loc8_2_1742410614837_250319_145654_hga"), # validation
                    self.patient.joinpath("Validation block_green_loc8_2_1742410614837_250319_145654_hga"), # test 
                ]
                        # Model parameters
                # CNNN
                self.trial_length = 400    
                self.num_kernels = 6#2
                self.spatial_kernel_size = 6
                self.temporal_kernel_size = 20 #40
                self.spatial_kernel_stride = 2
                self.temporal_kernel_stride = 9 #5
                self.spatial_kernel_padding =  0 #int(self.spatial_kernel_stride/2)
                self.temporal_kernel_padding =  0 #int(self.temporal_kernel_stride/2)

                # calculat post CNN shape
                self.xy_dim = int(( 33 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
                t_dim = int(( self.trial_length - self.temporal_kernel_size) / self.temporal_kernel_stride )  + 1
                self.in_channels: int = self.xy_dim**2 *self.num_kernels

                self.rnn_hidden_size: int = 640 #750
                self.n_features: int = 1
                self.n_labels: int = 2
                self.bidirectional: bool = True
                self.batch_size: int = 64
                self.rnn_layers: int = 2 #3
                
                self.cnn_dropout: float = 0.5 #0.6
                self.rnn_dropout: float = 0.3 #0.2
                
                # Training
                self.epochs: int = 30
                self.weight_decay: float = 1e-4 #2e-3
                self.learning_rate: float = 3e-4 #1e-4

            case 4:
                self.patient = Path(self.main_data_path + "/NSR-004-004").expanduser()
            
                self.recordings = [
                    self.patient.joinpath("center-out-easy-block1_FIH0068__1_1753286802143_250723_160642_hga"),
                    self.patient.joinpath("center-out-easy-block2_FIH0068__1_1753287064855_250723_161104_hga"),
                    self.patient.joinpath("center-out-easy-block3_FIH0068__3_1753287429688_250723_161709_hga"),
                    self.patient.joinpath("center-out-easy-block4_FIH0068__1_1753288012531_250723_162652_hga"), # validation
                    self.patient.joinpath("center-out-easy-block4_FIH0068__1_1753288012531_250723_162652_hga"), # test

                ]


           
                # Model parameters
                # CNNN
                self.trial_length = 400    
                self.num_kernels = 2
                self.spatial_kernel_size = 6
                self.temporal_kernel_size = 40
                self.spatial_kernel_stride = 2
                self.temporal_kernel_stride = 5
                self.spatial_kernel_padding =  0 #int(self.spatial_kernel_stride/2)
                self.temporal_kernel_padding =  0 #int(self.temporal_kernel_stride/2)

                # calculat post CNN shape
                self.xy_dim = int(( 33 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
                t_dim = int(( self.trial_length - self.temporal_kernel_size) / self.temporal_kernel_stride )  + 1
                self.in_channels: int = self.xy_dim**2 *self.num_kernels

                self.rnn_hidden_size: int = 750
                self.n_features: int = 1
                self.n_labels: int = 2
                self.bidirectional: bool = True
                self.batch_size: int = 64
                self.rnn_layers: int = 3
                
                self.cnn_dropout: float = 0.6
                self.rnn_dropout: float = 0.2
                
                # Training
                self.epochs: int = 50
                self.weight_decay: float = 2e-3
                self.learning_rate: float = 1e-4
            
                self.fs_hz = 100.0
                self.smooth_train_ms = 40.0
                self.smooth_val_ms = 100.0

            case 5:

                self.patient = Path(self.main_data_path + "/NSR-004-005").expanduser()
            
                self.recordings = [
                    self.patient.joinpath("train-veasy-hover_FIH0093_webgrid_2_1765990632034_251217_165712_hga"),
                    self.patient.joinpath("train-veasy-hover_FIH0093_webgrid_3_1765990947075_251217_170227_hga"),
                    self.patient.joinpath("train-veasy-hover_FIH0093_webgrid_4_1765991292272_251217_170812_hga"),
                    self.patient.joinpath("train-veasy-hover_FIH0093_webgrid_5_1765991611567_251217_171331_hga"), # validation
                    self.patient.joinpath("train-veasy-hover_FIH0093_webgrid_5_1765991611567_251217_171331_hga"), # test

                ]
                
                # Model parameters
                # CNNN
                self.trial_length = 400 #700    
                self.num_kernels = 8
                self.spatial_kernel_size = 9
                self.temporal_kernel_size = 20
                self.spatial_kernel_stride = 4
                self.temporal_kernel_stride = 3#1
                self.spatial_kernel_padding =  0 #int(self.spatial_kernel_stride/2)
                self.temporal_kernel_padding =  0 #int(self.temporal_kernel_stride/2)

                # calculat post CNN shape
                self.xy_dim = int(( 33 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
                t_dim = int(( self.trial_length - self.temporal_kernel_size) / self.temporal_kernel_stride )  + 1
                self.in_channels: int = self.xy_dim**2 *self.num_kernels

                self.rnn_hidden_size: int = 800
                self.n_features: int = 1
                self.n_labels: int = 2
                self.bidirectional: bool = True
                self.batch_size: int = 64
                self.rnn_layers: int = 6
                
                self.cnn_dropout: float = 0.64
                self.rnn_dropout: float = 0.15
                
                # Training
                self.epochs: int = 50
                self.weight_decay: float = 5e-4
                self.learning_rate: float = 4.5e-4
    
                self.smooth_train_ms = 40.0
                self.smooth_val_ms = 100.0

            case 102005:
                
                self.patient = Path(self.main_data_path + "/NSR-102-005").expanduser()
            

                self.recordings = [
                    self.patient.joinpath("webgrid-medium-random_FIH0117_train_1_1767103993233_251230_141313_hga"),
                    self.patient.joinpath("webgrid-medium-random_FIH0117_train_2_1767104299586_251230_141819_hga"),
                    self.patient.joinpath("webgrid-medium-random_FIH0117_train_3_1767104515800_251230_142155_hga"),
                    self.patient.joinpath("webgrid-medium-random_FIH0117_train_4_1767104746342_251230_142546_hga"), # validation
                    self.patient.joinpath("webgrid-medium-random_FIH0117_train_4_1767104746342_251230_142546_hga"), # test

                ]

                
                # Model parameters
                # CNNN
                self.trial_length = 350   
                self.num_kernels = 6
                self.spatial_kernel_size = 9
                self.temporal_kernel_size = 40
                self.spatial_kernel_stride = 1
                self.temporal_kernel_stride = 3
                self.spatial_kernel_padding =  0 #int(self.spatial_kernel_stride/2)
                self.temporal_kernel_padding =  0 #int(self.temporal_kernel_stride/2)

                # calculat post CNN shape
                self.xy_dim = int(( 33 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
                t_dim = int(( self.trial_length - self.temporal_kernel_size) / self.temporal_kernel_stride )  + 1
                self.in_channels: int = self.xy_dim**2 *self.num_kernels

                self.rnn_hidden_size: int = 384
                self.n_features: int = 1
                self.n_labels: int = 2
                self.bidirectional: bool = True
                self.batch_size: int = 64
                self.rnn_layers: int = 4
                
                self.cnn_dropout: float = 0.6
                self.rnn_dropout: float = 0.2
                
                # Training
                self.epochs: int = 100
                self.weight_decay: float = 4.5e-3
                self.learning_rate: float = 3e-4
            
            case 1020052:
                            
                    self.patient = Path(self.main_data_path + "/NSR-102-005").expanduser()
                

                    self.recordings = [
                        self.patient.joinpath("Session3", "webgrid-medium-random_FIH0117_train_3_1767117503028_251230_175823_hga"),
                        self.patient.joinpath("Session3","webgrid-medium-random_FIH0117_train_4_1767117701812_251230_180141_hga"),
                        self.patient.joinpath("Session3", "webgrid-medium-random_FIH0117_train_5_1767117907782_251230_180507_hga"),
                        self.patient.joinpath("Session4", "webgrid-easy_FIH0117_train_1_1767129389894_251230_211629_hga"),
                        self.patient.joinpath("Session4", "webgrid-easy_FIH0117_train_2_1767129551453_251230_211911_hga"),
                        self.patient.joinpath("Session4","webgrid-easy_FIH0117_train_3_1767129740140_251230_212220_hga"), # validation
                        self.patient.joinpath("Session5", "webgrid-medium_FIH0117_train_3_1767139362911_251231_000242_hga"),  # test

                    ]

                    
                    # Model parameters
                    # CNNN
                    self.trial_length = 350   
                    self.num_kernels = 6
                    self.spatial_kernel_size = 9
                    self.temporal_kernel_size = 40
                    self.spatial_kernel_stride = 1
                    self.temporal_kernel_stride = 3
                    self.spatial_kernel_padding =  0 #int(self.spatial_kernel_stride/2)
                    self.temporal_kernel_padding =  0 #int(self.temporal_kernel_stride/2)

                    # calculat post CNN shape
                    self.xy_dim = int(( 33 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
                    t_dim = int(( self.trial_length - self.temporal_kernel_size) / self.temporal_kernel_stride )  + 1
                    self.in_channels: int = self.xy_dim**2 *self.num_kernels

                    self.rnn_hidden_size: int = 384
                    self.n_features: int = 1
                    self.n_labels: int = 2
                    self.bidirectional: bool = True
                    self.batch_size: int = 64
                    self.rnn_layers: int = 4
                    
                    self.cnn_dropout: float = 0.6
                    self.rnn_dropout: float = 0.2
                    
                    # Training
                    self.epochs: int = 100
                    self.weight_decay: float = 4.5e-3
                    self.learning_rate: float = 3e-4
            case _:
                raise ValueError("Invalid participant")

                ## TODO -- parameterize x and y 

                self.trial_length = 400        
                # Model parameters
                # CNNN
                self.grid_dim_x =  33
                self.grid_dim_y =  33

                self.num_kernels = 2
                self.spatial_kernel_size = 6
                self.temporal_kernel_size = 40
                self.spatial_kernel_stride = 2
                self.temporal_kernel_stride = 5
                self.spatial_kernel_padding =  0 #int(self.spatial_kernel_stride/2)
                self.temporal_kernel_padding =  0 #int(self.temporal_kernel_stride/2)

                # calculat post CNN shape
                self.x_dim = int(( self.grid_dim_x - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
                self.y_dim = int(( self.grid_dim_y - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
                self.t_dim = int(( self.trial_length - self.temporal_kernel_size) / self.temporal_kernel_stride )  + 1
                self.in_channels: int = self.x_dim*self.y_dim *self.num_kernels

                self.rnn_hidden_size: int = 750
                self.n_features: int = 1
                self.n_labels: int = 2
                self.bidirectional: bool = True
                self.batch_size: int = 64
                self.rnn_layers: int = 3
                
                self.cnn_dropout: float = 0.6
                self.rnn_dropout: float = 0.2
                
                # Training
                self.epochs: int = 50#100
                self.weight_decay: float = 2e-3
                self.learning_rate: float = 1e-4
        if (use_general_params):
            self.trial_length: int = 400
            self.stride: int = 20
            self.batch_size: int = 64

            self.num_kernels: int = 6
            self.spatial_kernel_size: int = 6
            self.temporal_kernel_size: int = 1#20
            self.spatial_kernel_stride: int = 1#2
            self.temporal_kernel_stride: int = 9
            self.spatial_kernel_padding: int = 0
            self.temporal_kernel_padding: int = 0

            self.rnn_hidden_size: int = 640
            self.rnn_layers: int = 2
            self.bidirectional: bool = True
            self.cnn_dropout: float = 0.5
            self.rnn_dropout: float = 0.3

            self.epochs: int = 100
            self.learning_rate: float = 3e-4
            self.weight_decay: float = 1e-4

            self.xy_dim = int((32 - self.spatial_kernel_size) / self.spatial_kernel_stride) + 1
            self.in_channels = (self.xy_dim ** 2) * self.num_kernels
            self.n_labels: int = 2

            self.smooth_train_ms = 40.0
            self.smooth_val_ms = 100.0
            self.fs_hz = 100
    
    def update_spatial_kernel(self,spatial_kernel_size):
        self.spatial_kernel_size = spatial_kernel_size

        self.x_dim = int(( self.grid_dim_x - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
        self.y_dim = int(( self.grid_dim_x - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
        self.in_channels: int = self.x_dim*self.y_dim *self.num_kernels
    def update_num_kernels(self, num_kernels):
        self.num_kernels = num_kernels
        self.in_channels: int = self.x_dim*self.y_dim *self.num_kernels
    def update_grid_dims(self,x_dim, y_dim, spatial_kernel_size=None, spatial_kernel_stride=None ):
        self.grid_dim_x = x_dim
        self.grid_dim_y = y_dim

        if spatial_kernel_size is None:
            self.spatial_kernel_size = 1
        if spatial_kernel_stride is None:
            self.spatial_kernel_stride = 1

        # calculat post CNN shape
        self.x_dim = int(( self.grid_dim_x - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
        self.y_dim = int(( self.grid_dim_y - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
        self.in_channels: int = self.x_dim*self.y_dim *self.num_kernels
    def update_num_epochs(self, epochs):
        self.epochs = epochs
        print(f'Epochs updated to {self.epochs}')


class CNN2D_LSTM(nn.Module):
    """
    Bi‑LSTM whose inputs are (batch, time, channels, features) exactly
    like CRNN.  We first collapse (channels × features) with a Conv1d
    layer, then feed the result to a bidirectional LSTM.
    """

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        directions = 2 if config.bidirectional else 1

        self.input_dropout = nn.Dropout(config.cnn_dropout)


        self.preprocessing_conv = nn.Conv3d(
            in_channels = 1, 
            out_channels = config.num_kernels, # num kernels 
            kernel_size = (config.spatial_kernel_size, config.spatial_kernel_size, config.temporal_kernel_size),
            stride= (config.spatial_kernel_stride, config.spatial_kernel_stride, config.temporal_kernel_stride),  
            padding = (config.spatial_kernel_padding, config.spatial_kernel_padding, config.temporal_kernel_padding)  )  

    


        self.lstm = nn.LSTM(
            input_size=config.in_channels,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_layers,
            dropout=config.rnn_dropout,
            bidirectional=config.bidirectional )
        

        self.fc = nn.Linear(config.rnn_hidden_size * directions, config.n_labels)
        
    def forward(self, x, state=None):
        """
        Expects x.shape == (batch, time, channels, features)
        """
        # X: B, x, y, T F 
        B, Xd, Yd, T, F = x.shape

        #print(f'input shape {x.shape}')       #[64, 32, 32, 100, 1]
        
        #x = x.contiguous().moveaxis(2, 3).reshape(B, T, C * F).permute(0, 2, 1)
        
        x =x.contiguous().permute(0,4,1,2,3)
        #print(f'data resized to {x.shape}')

        
        x = self.input_dropout(x)  # dropout on raw inputs
        #print(f'data size after dropout {x.shape}')  # [64, 1, 32, 32, 100]
        
  
        # shape into conv layer (N,Cin,Din,Hin,Win) 
 
        x = self.preprocessing_conv(x)  # 3‑D conv
        #print(f'size after convolutions {x.shape}')
 
        x = torch.flatten(x, start_dim = 1,end_dim = 3)   # flatten out the x,y convs
        #print(f'shape after flattening {x.shape}')
        
        x = x.permute(2, 0, 1)  # shape -> (time, batch, features)
        #print(f'size going into LSTM {x.shape}')
        
        lstm_out, state = self.lstm(x, state)  # Bi‑LSTM
        #print(f'Out of LSTM {lstm_out.shape}')
        
        out = lstm_out[-1]  
        #print(f'out shape [out.shape]') 
        
        out = self.fc(out)  # final projection
        return out

class FrameDataset2D(Dataset):
    def __init__(self, frames_list, vxs_list, vys_list, trial_length = 100, stride = 20):
        self.windows = []  

        self.frames_list = frames_list
        self.vxs_list = [torch.tensor(v, dtype=torch.float32) for v in vxs_list]
        self.vys_list = [torch.tensor(v, dtype=torch.float32) for v in vys_list]
        self.trial_length = trial_length

        for i, frames in enumerate(frames_list):
            T = frames.shape[-1]
            for start in range(0, T - trial_length + 1, stride):
               self.windows.append((i, start))

        #print(f'Num windows {len(self.windows)}')
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        session_idx, start = self.windows[idx]
        end = start + self.trial_length

        x = self.frames_list[session_idx][:,:, start:end]
        vx = self.vxs_list[session_idx][start:end]
        vy = self.vys_list[session_idx][start:end]
        y = torch.stack([vx, vy], dim=0)

        return x, y

def get_dataloaders(config, x_set = [], y_set = []):
    main_data_path = config.main_data_path
    #main_data_path = '/home/becca/Data/PN_preprocessed/'
    #subject_num = 3
    electrode_x, electrode_y = electrode_map(probe_map_path=config.probe_map_path)
    
    patient = config.patient
    recordings = config.recordings

    #----------------------------
    # load training data 
    tr_frames = []  
    tr_vxs = []     
    tr_vys = [] 
    print("Loading Train Files...")  
    for idx, recording in enumerate(recordings[:-2]): 
        print(f"{recording}.pt")
        aligned_recording = torch.load( f"{recording}.pt", weights_only=False)
        frame, vx, vy = aligned_recording
        frame_t = len(frame)
        frame = torch.cat(frame, dim=1)
    
        ## reshape according to the map
        ordered_frame = torch.zeros(33,33,frame_t)
        
        for e in range(len(electrode_x)): 
            x = electrode_x[e]
            y = electrode_y[e]
        
            ordered_frame[x,y,:] = frame[e,:]
    
        if (config.fs_hz is not None and config.smooth_train_ms is not None and config.smooth_train_ms > 0):
            ordered_frame = gaussian_smooth_time(ordered_frame, fs_hz=config.fs_hz, sigma_ms=float(config.smooth_train_ms))
        if len(x_set) > 0:
            ordered_frame = ordered_frame[x_set][:,y_set]
        
        tr_frames.append(ordered_frame)
        tr_vxs.append(vx)
        tr_vys.append(vy)
    
    
    val_frames = [] 
    val_vxs = []     
    val_vys = []     
    print("Loading Val Files...")  
    print(f"{recordings[-2]}.pt")
    aligned_recording = torch.load( f"{recordings[-2]}.pt", weights_only=False)
    frame, vx, vy = aligned_recording
    frame_t = len(frame)
    frame = torch.cat(frame, dim=1)
    
    ## reshape according to the map
    ordered_frame = torch.zeros(33,33,frame_t)
    
    for e in range(len(electrode_x)): 
        x = electrode_x[e]
        y = electrode_y[e]
    
        ordered_frame[x,y,:] = frame[e,:]
    if (config.fs_hz is not None and config.smooth_val_ms is not None and config.smooth_val_ms > 0):
        ordered_frame = gaussian_smooth_time(ordered_frame, fs_hz=config.fs_hz, sigma_ms=float(config.smooth_val_ms))

    if len(x_set) > 0:
        ordered_frame = ordered_frame[x_set][:,y_set]
        
    val_frames.append(ordered_frame)
    val_vxs.append(vx)
    val_vys.append(vy)

    tst_frames = [] 
    tst_vxs = []     
    tst_vys = []     
    print("Loading Test Files...")  
    print(f"{recordings[-1]}.pt")
    aligned_recording = torch.load( f"{recordings[-1]}.pt", weights_only=False)
    frame, vx, vy = aligned_recording
    frame_t = len(frame)
    frame = torch.cat(frame, dim=1)
    
    ## reshape according to the map
    ordered_frame = torch.zeros(33,33,frame_t)
    
    for e in range(len(electrode_x)): 
        x = electrode_x[e]
        y = electrode_y[e]
    
        ordered_frame[x,y,:] = frame[e,:]
    if (config.fs_hz is not None and config.smooth_val_ms is not None and config.smooth_val_ms > 0):
        ordered_frame = gaussian_smooth_time(ordered_frame, fs_hz=config.fs_hz, sigma_ms=float(config.smooth_val_ms))

    if len(x_set) > 0:
        ordered_frame = ordered_frame[x_set][:,y_set]
        
    tst_frames.append(ordered_frame)
    tst_vxs.append(vx)
    tst_vys.append(vy)

    #----------
    batch_size = 64
    stride_length = 20
    
    CD_train = FrameDataset2D(tr_frames, tr_vxs, tr_vys, trial_length = config.trial_length, stride = stride_length)
    train_loader = DataLoader(CD_train, batch_size=batch_size, shuffle=True)
    
    CD_valid = FrameDataset2D(val_frames, val_vxs, val_vys, trial_length = config.trial_length, stride = stride_length)
    valid_loader = DataLoader(CD_valid, batch_size=batch_size, shuffle=True)
    
    CD_test = FrameDataset2D(tst_frames, tst_vxs, tst_vys, trial_length = config.trial_length, stride = stride_length)
    test_loader = DataLoader(CD_test, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader

def CNN3D_train(model, optimizer, criterion, config, train_loader, valid_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    n_epochs = config.epochs

    patience = config.patience
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
    
        for x, y in train_loader:
            #print(x.shape)
            
            # Rearrange y to [B, T, 2] to match model output
            y = y.permute(0, 2, 1)
            x = x.unsqueeze(-1)
    
            x = x.to(device).float()
            y = y.to(device).float()
    
            optimizer.zero_grad(set_to_none=True)
    
            # Forward pass, model should output [B, T, 2]
            pred_last = model(x)
            tgt_last = y[:, -1, :]
    
            # Single MSE loss on both vx and vy
            loss = criterion(pred_last, tgt_last)
    
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        avg_train_loss = running_loss / len(train_loader)
    
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.unsqueeze(-1) # adding a feature dim
                y = y.permute(0, 2, 1)

                x = x.to(device).float()
                y = y.to(device).float()

                tgt_last = y[:, -1, :]
                preds = model(x)
                val_loss += criterion(preds, tgt_last).item()
    
        avg_val_loss = val_loss / len(valid_loader)
        print(f'Epoch {epoch}/{n_epochs}: train:{avg_train_loss}  val:{avg_val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save best model
            torch.save(model.state_dict(), "best_model.pt")

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
    model.load_state_dict(torch.load("best_model.pt"))
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





