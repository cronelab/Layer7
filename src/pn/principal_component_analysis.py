from pn import CNN3D
from pn.CNN3D import *
import numpy as np
from sklearn.decomposition import PCA
from pn.input import *
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA
"""
def gaussian_smooth_time(x, fs_hz: float, sigma_ms: float = 40.0,
                         *, truncate: float = 3.0, fwhm: bool = False,
                         pad_mode: str = "reflect"):
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
            print('Smoothing')
            ordered_frame = gaussian_smooth_time(ordered_frame, fs_hz=config.fs_hz, sigma_ms=float(smooth_ms))
        if len(x_set) > 0:
            ordered_frame = ordered_frame[x_set][:,y_set]

        sessions.append((ordered_frame, vx, vy))
    return sessions

# load data

def get_pca(patient, recordings):
    all_electrode_frames = np.zeros((0, 1089))
    # Dimensions: (time, electrodes (xy flattened)) (Order changed for PCA/CCA)
    print(recordings[0][0].shape)
    np.transpose(recordings[0][0], (2, 0, 1))
    electrode_frames_xy_flattened = [np.array([frame.flatten() for frame in np.transpose(session[0], (2, 0, 1))]) for session in recordings]
    for session in electrode_frames_xy_flattened:
        # print(session.shape)
        # print(all_electrode_frames.shape)
        all_electrode_frames = np.concatenate((all_electrode_frames, session), axis=0)
    # print(training_frames)
    print(all_electrode_frames.shape)
    pca = PCA(n_components = 60)
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
        pca, transformed_data = get_pca(patient, original_sessions[patient] if patient != target_patient else original_sessions[patient][:-2])
        pcas[patient] = (pca, transformed_data)

    pca_t = pcas[target_patient][1]
    pca_t_all_sessions = np.concat([session for session in pca_t])
    ccas = {}
    for patient in non_target_patients:
        # Get the transformed latent space data for the source and target patients
        pca_s = pcas[patient][1]
        pca_s_all_sessions = np.concat([session for session in pca_s])

        cca = CCA(n_components=60)
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

            transformed_recording = pcas[target_patient][0].inverse_transform(latent_recording)
            reshaped_recording = np.transpose(np.reshape(transformed_recording, (transformed_recording.shape[0], 33, 33)), (1, 2, 0))

            aligned_recordings[patient].append(torch.tensor(reshaped_recording, dtype=torch.float32))

    # aligned_target_data = pcas[target_patient][0].inverse_transform(ccas[target_patient][1][0])

    print(f"Aligned target data is of shape {len(aligned_recordings)} by {len(aligned_recordings[non_target_patients[0]])} by {aligned_recordings[non_target_patients[0]][0].shape}")

    # ccas[0][0].transform(pca_transform, pcas[1][0].components_)

    val_sessions = [original_sessions[target_patient][-2]]
    test_sessions = [original_sessions[target_patient][-1]]

    return get_aligned_data_loader_v2(target_patient, aligned_recordings, original_sessions, val_sessions, test_sessions)

# original_records: Dict by patient number, contains list of sessions, sessions are tuples of (electrode frames, recording vxs, recording vys)
# aligned_patient_electrode_records
def get_aligned_data_loader_v2(target_patient: int, aligned_patient_electrode_records, original_records, val_sessions, test_sessions):
    config = Config(target_patient)

    train_sessions = original_records[target_patient][:-2]
    print(f"The shape of train_sessions is {len(train_sessions)} by {len(train_sessions[1])} by {len(train_sessions[1][1])}")
    train_frames, train_vxs, train_vys = [train_session[0] for train_session in train_sessions], [train_session[1] for train_session in train_sessions], [train_session[2] for train_session in train_sessions]

    val_frames, val_vxs, val_vys = [val_session[0] for val_session in val_sessions], [val_session[1] for val_session in val_sessions], [val_session[2] for val_session in val_sessions]

    test_frames, test_vxs, test_vys = [np.array(test_session[0]) for test_session in test_sessions], [np.array(test_session[1]) for test_session in test_sessions], [np.array(test_session[2]) for test_session in test_sessions]
    for patient in aligned_patient_electrode_records.keys():
        aligned_electrode_sessions = aligned_patient_electrode_records[patient]
        source_train_sessions = original_records[patient]
        source_train_vxs, source_train_vys = [train_session[1] for train_session in source_train_sessions], [train_session[2] for train_session in source_train_sessions]
        train_frames = train_frames + aligned_electrode_sessions
        train_vxs = train_vxs + source_train_vxs
        train_vys = train_vys + source_train_vys
    #----------
    batch_size = 64
    stride_length = 20

    # print(f"The shape of train_vxs is {len(train_vxs)} by {train_vxs.shape}")


    CD_train = FrameDataset2D(train_frames, train_vxs, train_vys, trial_length = config.trial_length, stride = stride_length)
    print(f"train frames has {len(CD_train.windows)} windows")
    # print(f"{CD_train.frames_list}, {CD_train.vxs_list}, {CD_train.vys_list}, {CD_train.windows}")
    print(f"The shape of train_frames is {len(train_frames)} by {[train_frame.shape for train_frame in train_frames]}")
    print(f"The shape of train_vxs is {len(train_vxs)} by {len(train_vxs[0])}")
    print(f"The shape of train_vys is {len(train_vys)} by {len(train_vys[0])}")
    train_loader = DataLoader(CD_train, batch_size=batch_size, shuffle=True)


    print(f"The shape of val_frames is {len(val_frames)} by {[val_frame.shape for val_frame in val_frames]}")
    print(f"The shape of val_vxs is {len(val_vxs)} by {len(val_vxs[0])}")
    print(f"The shape of val_vys is {len(val_vys)} by {len(val_vys[0])}")
    CD_valid = FrameDataset2D(val_frames, val_vxs, val_vys, trial_length = config.trial_length, stride = stride_length)
    print(f"val frames has {len(CD_valid.windows)} windows")
    # print(f"{CD_valid.frames_list}, {CD_valid.vxs_list}, {CD_valid.vys_list}, {CD_valid.windows}")
    valid_loader = DataLoader(CD_valid, batch_size=batch_size, shuffle=True)

    print(f"The shape of test_frames is {len(test_frames)} by {[test_frame.shape for test_frame in test_frames]}")
    print(f"The shape of test_vxs is {len(test_vxs)} by {len(test_vxs[0])}")
    print(f"The shape of test_vys is {len(test_vys)} by {len(test_vys[0])}")
    CD_test = FrameDataset2D(test_frames, test_vxs, test_vys, trial_length = config.trial_length, stride = stride_length)
    test_loader = DataLoader(CD_test, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader
















