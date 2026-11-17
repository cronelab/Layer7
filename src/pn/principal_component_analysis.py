from pn import CNN3D
from pn.CNN3D import *
import numpy as np
from torch import nn
from sklearn.decomposition import PCA
from pn.input import *
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA
"""
def gaussian_smooth_time(x, fs_hz: float, sigma_ms: float = 40.0,
                         *, truncate: float = 3.0, fwhm: bool = False,
                         pad_mode:  str = "reflect"):
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
"""

features = 60

class CNN_LSTM(nn.Module):
    """
    Bi‑LSTM whose inputs are (batch, time, channels, features) exactly
    like CRNN.  We first collapse (channels × features) with a Conv1d
    layer, then feed the result to a bidirectional LSTM.
    """

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        print(f"Config: {config}")
        directions = 2 if config.bidirectional else 1

        self.input_dropout = nn.Dropout(config.cnn_dropout)

        print(f"Init Convolution Params: (spatial_kernel_stride: {config.spatial_kernel_stride}, spatial_kernel_size: {config.spatial_kernel_size}, num_kernels: {config.num_kernels}, xy_dim: {config.xy_dim}")
        # print(f"Expected xy_dim: {int(( 33 - config.spatial_kernel_size) / config.spatial_kernel_stride )  + 1}")
        expected_xy_dim = int(( 33 - config.spatial_kernel_size) / config.spatial_kernel_stride )  + 1
        # print(f"Expected in_channels: {expected_xy_dim**2 * config.num_kernels}")
        self.preprocessing_conv = nn.Conv1d(
            in_channels = features,
            out_channels = config.num_kernels, # num kernels
            kernel_size = config.temporal_kernel_size,
            stride= config.temporal_kernel_stride,
            padding = config.temporal_kernel_padding)



        print(f"Init LSTM Params: {config.in_channels}")
        self.lstm = nn.LSTM(
            input_size=config.num_kernels,
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
        B, Xd, T, F = x.shape

        # print(f'input shape {x.shape}')       #[64, 60, 100, 1]

        #x = x.contiguous().moveaxis(2, 3).reshape(B, T, C * F).permute(0, 2, 1)

        x =x.contiguous().permute(0,3,1,2)
        #print(f'data resized to {x.shape}')


        x = self.input_dropout(x)  # dropout on raw inputs
        # print(f'data size after dropout {x.shape}')  # [64, 1, 32, 32, 100]


        # shape into conv layer (N,Cin,Din,Hin,Win)

        x = torch.flatten(x, start_dim = 1,end_dim = 2)   # flatten out the x,y convs
        # print(f'shape after flattening {x.shape}')
        x = self.preprocessing_conv(x)  # 1‑D conv
        # print(f'size after convolutions {x.shape}')

        # x = torch.flatten(x, start_dim = 1,end_dim = 2)   # flatten out the x,y convs
        #print(f'shape after flattening {x.shape}')

        x = x.permute(2, 0, 1)  # shape -> (time, batch, features)
        # print(f'size going into LSTM {x.shape}')
        lstm_out, state = self.lstm(x, state)  # Bi‑LSTM
        #print(f'Out of LSTM {lstm_out.shape}')

        out = lstm_out[-1]
        #print(f'out shape [out.shape]')

        out = self.fc(out)  # final projection
        return out

class FrameDataset1D(Dataset):
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

        x = self.frames_list[session_idx][:, start:end]
        vx = self.vxs_list[session_idx][start:end]
        vy = self.vys_list[session_idx][start:end]
        y = torch.stack([vx, vy], dim=0)

        return x, y

def get_recordings(config, recordings, smooth_ms,  x_set = [], y_set = []):
    sessions = []

    electrode_x, electrode_y = electrode_map(probe_map_path=config.probe_map_path)

    for recording in recordings:
        print(f"{recording}.pt")
        aligned_recording = torch.load( f"{recording}.pt", weights_only=False)
        frame, vx, vy = aligned_recording
        frame_t = len(frame)
        frame = torch.cat(frame, dim=1)

        ## reshape according to the map
        ordered_frame = torch.zeros(33, 33, frame_t)

        for e in range(len(electrode_x)):
            electrode_x_position = electrode_x[e]
            electrode_y_position = electrode_y[e]

            ordered_frame[electrode_x_position, electrode_y_position, :] = frame[e,:]

        if (config.fs_hz is not None and smooth_ms is not None and smooth_ms > 0):
            ordered_frame = gaussian_smooth_time(ordered_frame, fs_hz=config.fs_hz, sigma_ms=float(smooth_ms))
        if len(x_set) > 0:
            ordered_frame = ordered_frame[x_set][:,y_set]

        sessions.append((ordered_frame, vx, vy))
    return sessions

# load data

def get_pca(patient, recordings, fit_end_index):
    all_electrode_frames = np.zeros((0, 1089))
    # Dimensions: (time, electrodes (xy flattened)) (Order changed for PCA/CCA)
    print(recordings[0][0].shape)
    np.transpose(recordings[0][0], (2, 0, 1))
    electrode_frames_xy_flattened = [np.array([frame.flatten() for frame in np.transpose(session[0], (2, 0, 1))]) for session in recordings]
    for session in electrode_frames_xy_flattened[0:fit_end_index]:
        # print(session.shape)
        # print(all_electrode_frames.shape)
        all_electrode_frames = np.concatenate((all_electrode_frames, session), axis=0)
    # print(training_frames)
    print(all_electrode_frames.shape)
    pca = PCA(n_components = features)
    pca.fit(all_electrode_frames)
    # print(pca.explained_variance_ratio_)
    print(f"{pca.components_.shape} components for patient {patient}, total variance explained: {sum(pca.explained_variance_ratio_) * 100}%")
    # print(pca.singular_values_)

    return pca, [pca.transform(session_electrodes) for session_electrodes in electrode_frames_xy_flattened]

def get_aligned_data_loader(target_patient, all_patients):
    pcas = {}
    original_sessions = {}
    non_target_patients = list(filter(lambda patient: patient != target_patient, all_patients))
    print(non_target_patients)

    for patient in all_patients:
        config = CNN3D.Config(patient)
        original_sessions[patient] = get_recordings(config, config.recordings, CNN3D.Config(target_patient).smooth_train_ms)

    for patient in all_patients:
        pca, transformed_data = get_pca(patient, original_sessions[patient], None if patient != target_patient else -2)
        pcas[patient] = (pca, transformed_data)

    pca_t = pcas[target_patient][1]
    pca_t_all_sessions = np.concat([session for session in pca_t])
    ccas = {}
    for patient in non_target_patients:
        # Get the transformed latent space data for the source and target patients
        pca_s = pcas[patient][1]
        pca_s_all_sessions = np.concat([session for session in pca_s])

        cca = CCA(n_components = features)
        min_length = min(len(pca_t_all_sessions), len(pca_s_all_sessions))
        cca.fit(pca_s_all_sessions[:min_length], pca_t_all_sessions[:min_length])

        print(f"The shape of the target patient's PCA is {len(pca_t_all_sessions)} by {len(pca_t_all_sessions[0])}, the shape of patient {patient}'s is {len(pca_s_all_sessions)} by {len(pca_s_all_sessions[0])}")

        pca_s_c = [cca.transform(session) for session in pca_s]
        print(f"The shape Patient {patient}'s CCA transformed data is {len(pca_s_c)} by {len(pca_s_c[0])} by {len(pca_s_c[0][0])}")
        ccas[patient] = (cca, pca_s_c)

    print(len(pcas[target_patient][1][0]))

    # pca_transform = pcas[0][0].transform(pcas[0][1])
    # print(len(pca_transform[0]))

    aligned_recordings = {}

    for patient in non_target_patients:
        _, latent_recordings = ccas[patient]
        aligned_recordings[patient] = []
        for latent_recording in latent_recordings:

            # transformed_recording = pcas[target_patient][0].inverse_transform(latent_recording)
            # reshaped_recording = np.transpose(np.reshape(transformed_recording, (transformed_recording.shape[0], 33, 33)), (1, 2, 0))

            aligned_recordings[patient].append(torch.tensor(np.transpose(latent_recording, (1, 0)), dtype=torch.float32))
    aligned_recordings[target_patient] = []
    for latent_recording in pcas[target_patient][1][:-2]:
        aligned_recordings[target_patient].append(torch.tensor(np.transpose(latent_recording, (1, 0)), dtype=torch.float32))
    # aligned_recordings[target_patient] = pcas[target_patient][0]

    # aligned_target_data = pcas[target_patient][0].inverse_transform(ccas[target_patient][1][0])

    print(f"Aligned target data is of shape {len(aligned_recordings)} by {len(aligned_recordings[non_target_patients[0]])} by {aligned_recordings[non_target_patients[0]][0].shape}")

    # ccas[0][0].transform(pca_transform, pcas[1][0].components_)

    val_sessions = [torch.tensor(np.transpose(pcas[target_patient][1][-1], (1, 0)), dtype=torch.float32)]
    test_sessions = [torch.tensor(np.transpose(pcas[target_patient][1][-2], (1, 0)), dtype=torch.float32)]

    return get_aligned_data_loader_v2(target_patient, aligned_recordings, original_sessions, val_sessions, test_sessions)

# original_records: Dict by patient number, contains list of sessions, sessions are tuples of (electrode frames, recording vxs, recording vys)
# aligned_patient_electrode_records
def get_aligned_data_loader_v2(target_patient: int, aligned_patient_electrode_records, original_records, val_sessions, test_sessions):
    config = Config(target_patient)

    # train_sessions = [] # original_records[target_patient][:-2]
    # print(f"The shape of train_sessions is {len(train_sessions)} by {len(train_sessions[1])} by {len(train_sessions[1][1])}")
    train_frames, train_vxs, train_vys = [], [], []# [train_session[0] for train_session in train_sessions], [train_session[1] for train_session in train_sessions], [train_session[2] for train_session in train_sessions]

    val_frames, val_vxs, val_vys = val_sessions, [original_records[target_patient][-2][1]], [original_records[target_patient][-2][2]]

    test_frames, test_vxs, test_vys = test_sessions, [original_records[target_patient][-1][1]], [original_records[target_patient][-1][2]]
    for patient in aligned_patient_electrode_records.keys():
        aligned_electrode_sessions = aligned_patient_electrode_records[patient]
        source_train_sessions = original_records[patient] if patient != target_patient else original_records[patient][:-2]
        source_train_vxs, source_train_vys = [train_session[1] for train_session in source_train_sessions], [train_session[2] for train_session in source_train_sessions]
        train_frames = train_frames + aligned_electrode_sessions
        train_vxs = train_vxs + source_train_vxs
        train_vys = train_vys + source_train_vys
    #----------
    batch_size = 64
    stride_length = 20

    # print(f"The shape of train_vxs is {len(train_vxs)} by {train_vxs.shape}")


    CD_train = FrameDataset1D(train_frames, train_vxs, train_vys, trial_length = config.trial_length, stride = stride_length)
    print(f"train frames has {len(CD_train.windows)} windows")
    # print(f"{CD_train.frames_list}, {CD_train.vxs_list}, {CD_train.vys_list}, {CD_train.windows}")
    print(f"The shape of train_frames is {len(train_frames)} by {[train_frame.shape for train_frame in train_frames]}")
    print(f"The shape of train_vxs is {len(train_vxs)} by {[len(train_vxs_one) for train_vxs_one in train_vxs]}")
    print(f"The shape of train_vys is {len(train_vys)} by {[len(train_vys_one) for train_vys_one in train_vys]}")
    train_loader = DataLoader(CD_train, batch_size=batch_size, shuffle=True)


    print(f"The shape of val_frames is {len(val_frames)} by {[val_frame.shape for val_frame in val_frames]}")
    print(f"The shape of val_vxs is {len(val_vxs)} by {[len(val_vxs_one) for val_vxs_one in val_vxs]}")
    print(f"The shape of val_vys is {len(val_vys)} by {[len(val_vys_one) for val_vys_one in val_vys]}")
    CD_valid = FrameDataset1D(val_frames, val_vxs, val_vys, trial_length = config.trial_length, stride = stride_length)
    print(f"val frames has {len(CD_valid.windows)} windows")
    # print(f"{CD_valid.frames_list}, {CD_valid.vxs_list}, {CD_valid.vys_list}, {CD_valid.windows}")
    valid_loader = DataLoader(CD_valid, batch_size=batch_size, shuffle=True)

    print(f"The shape of test_frames is {len(test_frames)} by {[test_frame.shape for test_frame in test_frames]}")
    print(f"The shape of test_vxs is {len(test_vxs)} by {len(test_vxs[0])}")
    print(f"The shape of test_vys is {len(test_vys)} by {len(test_vys[0])}")
    CD_test = FrameDataset1D(test_frames, test_vxs, test_vys, trial_length = config.trial_length, stride = stride_length)
    test_loader = DataLoader(CD_test, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader
















