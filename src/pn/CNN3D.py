
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class Config:
    def __init__(self, subject_num, use_general_params = False,main_data_path = 'D:\\Becca\\Data\\PN_preprocessed'):
        # experiment parameters 

        # Subject Info
        #self.main_data_path = '/home/becca/Data/PN_preprocessed/'  # on corticom training
        self.main_data_path = main_data_path # on lab windows box
        
        self.smooth_train_ms = None
        self.smooth_val_ms = None
        self.fs_hz = None
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
                self.xy_dim = int(( 32 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
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
                self.xy_dim = int(( 32 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
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
                self.xy_dim = int(( 32 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
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
                self.xy_dim = int(( 32 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
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
                self.xy_dim = int(( 32 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
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
            
            case 102005411:
                self.patient = Path(self.main_data_path + "/NSR-102-005").expanduser()

                '''
                recording_ids = [key for key in p5u.train_block_id.keys()]
                train_len = int(np.floor(0.8*len(recording_ids)))
                train_ids = recording_ids[:train_len]
                recording_ids = train_ids
                '''
                #recording_ids = [key for key in p5u.train_block_id.keys()]

                #recording_ids = [175,176, 177, 188, 189, 190, 203,211,211] # hpo on
                recording_ids = [175,176, 177, 188, 189, 190, 203]  #hpo on 6/25
                self.recordings = []
                for id in recording_ids:
                    filepath = p5u.train_block_id[id].split("\\")
                    if (os.path.exists(self.patient.joinpath(filepath[0], filepath[-1]+"_hga.pt"))):
                        self.recordings.append(
                            self.patient.joinpath(filepath[0], filepath[-1]+"_hga") )

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
                self.xy_dim = int(( 32 - self.spatial_kernel_size) / self.spatial_kernel_stride )  + 1
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
                self.epochs: int = 10
                self.weight_decay: float = 4.5e-3
                self.learning_rate: float = 3e-4
            case _:
                raise ValueError("Invalid participant")

                ## TODO -- parameterize x and y 

                self.trial_length = 400        
                # Model parameters
                # CNNN
                self.grid_dim_x =  32
                self.grid_dim_y =  32

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

def CNN3D_train(model, optimizer, criterion, config, train_loader, valid_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    n_epochs = config.epochs
    print(n_epochs, "epochs")
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
    
        for x, y in train_loader:
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




