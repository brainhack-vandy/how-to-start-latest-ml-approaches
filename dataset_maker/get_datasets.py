import mne
import os
import json

import numpy as np
import pandas as pd
import re

from dataset_maker import preproc
from scipy.signal import butter, filtfilt
import torch
import math
import pickle



def download_vu_dataset(sub, scan, path_to_dataset, output_eegmneraw=False, out_eegchan='EEG'):
    """
    Load VU resting-state EEG and fMRI data

    ------
    Input:

    patient : str
         example - 'trio1'
    path_to_dataset
         exampel - ../data/eyes_open_closed_dataset

    return
    df_eeg, df_fmri, labels
    """

    # todo: change "patient" here to be something like "sub", "scan"...
    patient = 'sub' + '{:02d}'.format(int(sub)) + '-scan' + '{:02d}'.format(int(scan))
    eeg_path_set_file = os.path.join(path_to_dataset, 'EEG_set', f'{patient}_eeg_pp.set')
    fMRI_path_difumo = os.path.join(path_to_dataset, 'fMRI_difumo', f'{patient}_difumo_roi.pkl')
    new_fps = 200

    # ----- Load and preprocess EEG -----
    raw = mne.io.read_raw_eeglab(eeg_path_set_file)
    original_fps = raw.info['sfreq']  # original_fps = 250
    if original_fps != new_fps:
        raw.resample(new_fps)
    if len(raw.ch_names) > 32:
        vector_exclude = ['EOG1', 'EOG2', 'EMG1', 'EMG2', 'EMG3', 'ECG',
                          'CWL1', 'CWL2', 'CWL3', 'CWL4']
    else:
        vector_exclude = ['EOG1', 'EOG2', 'EMG1', 'EMG2', 'EMG3', 'ECG']

    if output_eegmneraw == True:
        if out_eegchan == 'EEG':
            raw.drop_channels(vector_exclude)
        elif out_eegchan == 'EMG':
            raw.pick_channels(['EMG1', 'EMG2', 'EMG3'])
        df_eeg = raw
    else:
        df_eeg = raw.to_data_frame()
        df_eeg = df_eeg.drop(vector_exclude, axis=1)

    # ----- Load fMRI ROI time series (Difumo atlas) -----
    df_fmri = pd.read_pickle(fMRI_path_difumo)
    ROI_labels = df_fmri.columns.to_list()

    return df_eeg, df_fmri, ROI_labels


def convert_to_tensor(data):
    tensors = [torch.tensor(arr) for arr in data]
    return torch.stack(tensors, dim=0)


def prepare_onesub_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = args.dataset_root
    tr = args.TR
    event_sync_name = args.mri_sync_event

    tmin = -16
    tmax = 0
    crop = 3200  # 16 * 200

    sub_ind, scan_ind = re.findall(r'\d+', args.dataname)
    sub_ind, scan_ind = int(sub_ind), int(scan_ind)

    eeg_raw, fmri, labels_roi_full = download_vu_dataset(sub_ind, scan_ind, root, output_eegmneraw=True)
    fmri_np = fmri.to_numpy().T
    # extract ROI

    df = pd.DataFrame(data=fmri_np.T, columns=labels_roi_full)
    df_filter = df[args.labels_roi]
    fmri_np = df_filter.to_numpy().T
    if len(fmri_np.shape) < 2:
        fmri_np = fmri_np[np.newaxis, ...]

    # Optional: add filtering, filter below 0.15Hz
    # This part can also be moved to making_difumo.py
    fs = 1 / tr
    nyquist = 0.5 * fs
    low = 0.15 / nyquist  # Low cutoff frequency (as a fraction of Nyquist frequency)
    b, a = butter(N=5, Wn=low, btype='low', analog=False)  # Using a 5th order filter
    # Apply the filter
    fmri_np = filtfilt(b, a, fmri_np, axis=1)

    fmri_norm, _ = preproc.normalize_data(fmri_np)

    # epoching data, and divide them into train, val and test
    data_epoch, eeg_info = preproc.epoching_seq2one(eeg_raw, fmri_norm,
                                                    tmin, tmax, event_sync_name, ifnorm=0,
                                                    crop=crop)
    traincrop = int(0.8 * len(data_epoch["eeg"]))
    valcrop = int(0.1 * len(data_epoch["eeg"])) + traincrop

    eeg_train = data_epoch["eeg"][:traincrop]
    fmri_train = data_epoch["fmri"][:traincrop]

    # consider the auto-correlaiton of fMRI, preventing data leakage
    t_overlap = 20 if abs(tmin) <= 20 else abs(tmin)  # Length of HRF consideration
    N_overlap = math.ceil(t_overlap / tr)

    traincrop += N_overlap
    eeg_val = data_epoch["eeg"][traincrop:valcrop]
    fmri_val = data_epoch["fmri"][traincrop:valcrop]

    valcrop += N_overlap
    eeg_test = data_epoch["eeg"][valcrop:]
    fmri_test = data_epoch["fmri"][valcrop:]

    # convert to tensor
    # eeg_train_tensor = [torch.tensor(arr) for arr in eeg_train]
    # fmri_train_tensor = [torch.tensor(arr) for arr in fmri_train]
    # eeg_val_tensor = [torch.tensor(arr) for arr in eeg_val]
    # fmri_val_tensor = [torch.tensor(arr) for arr in fmri_val]
    # eeg_test_tensor = [torch.tensor(arr) for arr in eeg_test]
    # fmri_test_tensor = [torch.tensor(arr) for arr in fmri_test]
    #
    # eeg_train_tensor = torch.stack(eeg_train_tensor, dim=0)
    # eeg_val_tensor = torch.stack(eeg_val_tensor, dim=0)
    # eeg_test_tensor = torch.stack(eeg_test_tensor, dim=0)
    #
    # fmri_train_tensor = torch.stack(fmri_train_tensor, dim=0).squeeze()
    # fmri_val_tensor = torch.stack(fmri_val_tensor, dim=0).squeeze()
    # fmri_test_tensor = torch.stack(fmri_test_tensor, dim=0).squeeze()

    eeg_train_tensor = convert_to_tensor(eeg_train)
    eeg_val_tensor = convert_to_tensor(eeg_val)
    eeg_test_tensor = convert_to_tensor(eeg_test)

    fmri_train_tensor = convert_to_tensor(fmri_train).squeeze()
    fmri_val_tensor = convert_to_tensor(fmri_val).squeeze()
    fmri_test_tensor = convert_to_tensor(fmri_test).squeeze()

    train_dataset = torch.utils.data.TensorDataset(eeg_train_tensor, fmri_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(eeg_val_tensor, fmri_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(eeg_test_tensor, fmri_test_tensor)

    return train_dataset, test_dataset, val_dataset


def prepare_full_dataloader(args):
    if args.prepro_datapath:
        # with open(args.prepro_datapath, 'rb') as f:
        #     train, val, test, args_data = pickle.load(f)
        #
        # eeg_train_tensor, fmri_train_tensor = train[0], train[1]
        # eeg_test_tensor, fmri_test_tensor = test[0], test[1]
        # eeg_val_tensor, fmri_val_tensor = val[0], val[1]
        #
        # train_dataset = torch.utils.data.TensorDataset(eeg_train_tensor, fmri_train_tensor)
        # val_dataset = torch.utils.data.TensorDataset(eeg_val_tensor, fmri_val_tensor)
        # test_dataset = torch.utils.data.TensorDataset(eeg_test_tensor, fmri_test_tensor)

        # Load preprocessed data
        try:
            with open(args.prepro_datapath, 'rb') as file:
                train_data, val_data, test_data, args_data = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {args.prepro_datapath}")
        except Exception as e:
            raise RuntimeError(f"Error loading data from {args.prepro_datapath}: {e}")

        # Unpack data tensors
        eeg_train_tensor, fmri_train_tensor = train_data
        eeg_val_tensor, fmri_val_tensor = val_data
        eeg_test_tensor, fmri_test_tensor = test_data

        # Create TensorDatasets
        train_dataset = torch.utils.data.TensorDataset(eeg_train_tensor, fmri_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(eeg_val_tensor, fmri_val_tensor)
        test_dataset = torch.utils.data.TensorDataset(eeg_test_tensor, fmri_test_tensor)

        # Optional: Log dataset creation
        print("Datasets successfully created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        return train_dataset, test_dataset, val_dataset

    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = args.dataset_root
    tr = args.TR
    event_sync_name = args.mri_sync_event

    tmin = -16
    tmax = 0
    crop = 3200

    # scan_todo_file = 'C:\\yamin\\eeg-fmri-DL\\EEG2fMRI-seq2one\\scan_to_do_finalversion.xlsx'
    scan_sheet = args.split_index_sheet
    sheet_name = 'fold1'
    df = pd.read_excel(io=scan_sheet, sheet_name=sheet_name)
    scans = df['scan_name'].values
    train_ind = np.where(df['train'].values == 1)[0]
    val_ind = np.where(df['val'].values == 1)[0]
    test_ind = np.where(df['test'].values == 1)[0]

    # eeg_train, fmri_train = [], []
    # eeg_val, fmri_val = [], []
    # eeg_test, fmri_test = [], []

    # Data containers
    eeg_data = {"train": [], "val": [], "test": []}
    fmri_data = {"train": [], "val": [], "test": []}

    for ss in range(len(scans)):
        sub_ind, scan_ind = re.findall(r'\d+', scans[ss])
        print('Start preparing', scans[ss], '...')
        sub_ind, scan_ind = int(sub_ind), int(scan_ind)
        eeg_raw, fmri, labels_roi_full = download_vu_dataset(sub_ind, scan_ind, root, output_eegmneraw=True)

        eeg_raw.load_data()
        eeg_raw.filter(l_freq=0.5, h_freq=None)
        fmri_np = fmri.to_numpy().T

        # extract ROI
        df = pd.DataFrame(data=fmri_np.T, columns=labels_roi_full)
        df_filter = df[args.labels_roi]
        fmri_np = df_filter.to_numpy().T
        if len(fmri_np.shape) < 2:
            fmri_np = fmri_np[np.newaxis, ...]

        # Optional: filter fMRI below 0.15Hz, can be removed if already done in preprocessing
        fs = 1 / tr
        nyquist = 0.5 * fs
        low = 0.15 / nyquist  # Low cutoff frequency (as a fraction of Nyquist frequency)
        b, a = butter(N=5, Wn=low, btype='low', analog=False)  # Using a 5th order filter

        # Apply the filter
        fmri_np = filtfilt(b, a, fmri_np, axis=1)

        fmri_norm, _ = preproc.normalize_data(fmri_np)

        # epoching data, and divide them into train, val and test
        data_epoch, eeg_info = preproc.epoching_seq2one(eeg_raw, fmri_norm,
                                                        tmin, tmax, event_sync_name, ifnorm=0,
                                                        crop=crop)

        # if ss in train_ind:
        #     eeg_train = eeg_train + data_epoch["eeg"]
        #     fmri_train = fmri_train + data_epoch["fmri"]
        # elif ss in val_ind:
        #     eeg_val = eeg_val + data_epoch["eeg"]
        #     fmri_val = fmri_val + data_epoch["fmri"]
        # elif ss in test_ind:
        #     eeg_test = eeg_test + data_epoch["eeg"]
        #     fmri_test = fmri_test + data_epoch["fmri"]
        # Assign to train, val, or test

        if ss in train_ind:
            eeg_data["train"] += data_epoch["eeg"]
            fmri_data["train"] += data_epoch["fmri"]
        elif ss in val_ind:
            eeg_data["val"] += data_epoch["eeg"]
            fmri_data["val"] += data_epoch["fmri"]
        elif ss in test_ind:
            eeg_data["test"] += data_epoch["eeg"]
            fmri_data["test"] += data_epoch["fmri"]

    # convert to tensor
    eeg_tensors = {key: convert_to_tensor(eeg_data[key]) for key in eeg_data}
    fmri_tensors = {key: convert_to_tensor(fmri_data[key]).squeeze() for key in fmri_data}

    # Package data
    train_data = eeg_tensors["train"], fmri_tensors["train"]
    val_data = eeg_tensors["val"], fmri_tensors["val"]
    test_data = eeg_tensors["test"], fmri_tensors["test"]

    # eeg_train_tensor = [torch.tensor(arr) for arr in eeg_train]
    # fmri_train_tensor = [torch.tensor(arr) for arr in fmri_train]
    # eeg_val_tensor = [torch.tensor(arr) for arr in eeg_val]
    # fmri_val_tensor = [torch.tensor(arr) for arr in fmri_val]
    # eeg_test_tensor = [torch.tensor(arr) for arr in eeg_test]
    # fmri_test_tensor = [torch.tensor(arr) for arr in fmri_test]
    #
    # eeg_train_tensor = torch.stack(eeg_train_tensor, dim=0)
    # eeg_val_tensor = torch.stack(eeg_val_tensor, dim=0)
    # eeg_test_tensor = torch.stack(eeg_test_tensor, dim=0)
    #
    # fmri_train_tensor = torch.stack(fmri_train_tensor, dim=0).squeeze()
    # fmri_val_tensor = torch.stack(fmri_val_tensor, dim=0).squeeze()
    # fmri_test_tensor = torch.stack(fmri_test_tensor, dim=0).squeeze()

    # train_data = eeg_train_tensor, fmri_train_tensor
    # val_data = eeg_val_tensor, fmri_val_tensor
    # test_data = eeg_test_tensor, fmri_test_tensor

    # Optional: save dataset here for saving time later
    if args.save_input_tensor:
        # ROI Name Mapping
        roi_map = {
            "Middle frontal gyrus anterior": "Mid_fron_gyr_ant",
            "Heschl’s gyrus": "Hesch",
            "global signal clean": "glb"
        }
        roi_name = roi_map.get(args.labels_roi, args.labels_roi)

        # if args.labels_roi == "Middle frontal gyrus anterior":
        #     roi_name = "Mid_fron_gyr_ant"
        # elif args.labels_roi == "Heschl’s gyrus":
        #     roi_name = "Hesch"
        # elif args.labels_roi == "global signal clean":
        #     roi_name = "glb"
        # else:
        #     roi_name = args.labels_roi
        with open(f'vu_seq2one_full_16s_{roi_name}.pkl', 'wb') as f:
            pickle.dump([train_data, val_data, test_data, args], f)

    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(*train_data)
    val_dataset = torch.utils.data.TensorDataset(*val_data)
    test_dataset = torch.utils.data.TensorDataset(*test_data)

    return train_dataset, test_dataset, val_dataset
