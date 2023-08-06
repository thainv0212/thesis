import os
import torch
import torchaudio
import json
import numpy as np
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from functools import lru_cache


def unstack_tensor(tensor, dim=0):
    tensor_lst = []
    for i in range(tensor.size(dim)):
        tensor_lst.append(tensor[i])
    tensor_unstack = torch.cat(tensor_lst, dim=0)
    return tensor_unstack


class MHDDataset(Dataset):
    def __init__(self, data_file, train=True):

        self.train = train
        if train:
            self.data_file = os.path.join(data_file, "mhd_train.pt")
        else:
            self.data_file = os.path.join(data_file, "mhd_test.pt")

        if not os.path.exists(data_file):
            raise RuntimeError(
                'MHD Dataset not found. Please generate dataset and place it in the data folder.')

        # Load data
        # import ipdb; ipdb.set_trace()
        self._label_data, self._image_data, self._trajectory_data, self._sound_data, self._traj_normalization, self._sound_normalization = torch.load(
            self.data_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            array: image_data, sound_data, trajectory_data, label_data
        """
        audio = unstack_tensor(self._sound_data[index]).unsqueeze(0)
        audio_perm = audio.permute(0, 2, 1)
        # import ipdb; ipdb.set_trace()
        return self._image_data[index], audio_perm, self._trajectory_data[index], self._label_data[index]

    def __len__(self):
        return len(self._label_data)

    def get_audio_normalization(self):
        return self._audio_normalization

    def get_traj_normalization(self):
        return self._traj_normalization


class SoundDataset3(Dataset):
    def __init__(self, root_dir=None, files=None, transforms=None, frame_col='frameDataNonDelay', audio_col='rawAudio',
                 batch_size=32, sequence_len=20):
        self.root_dir = root_dir
        self.transforms = transforms
        self.frame_col = frame_col
        self.audio_col = audio_col
        # if self.root_dir is not None:
        #     self.files = [f for f in os.listdir(root_dir) if '.txt' in f]
        if root_dir is not None:
            self.files = np.array(
                [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if 'logdata_' in f])
        elif files is not None:
            self.files = np.array(files)
        self.batch_size = batch_size
        print('load data to get data size')
        self.data_size = 0
        self.data_size_batch = []

        self.sequence_len = sequence_len
        # self.current_file = None
        # self.current_file_data = None
        # self.files_data = {k: json.load(open(k, 'r')) for k in self.files}
        self.files_data = {}
        for k in tqdm(self.files):
            self.files_data[k] = json.load(open(k, 'r'))
        for file_data in self.files_data.values():
            self.data_size_batch.append(self.data_size)
            self.data_size += len(file_data)
        self.data_size_batch = np.array(self.data_size_batch)
        print('data size', self.data_size)
        # self.files = self.files[:1]

    def preprocess(self, text):
        # data = json.loads(text)
        # data_total = []
        # for l in text:
        data = {'Audio': text[self.audio_col], 'Frame': text[self.frame_col]}
        if self.transforms:
            if 'all' in self.transforms.keys():
                for col in data.keys():
                    if col != 'id' and col != 'file':
                        data[col] = self.transforms['all'](data[col])
            for k, f in self.transforms.items():
                if k != 'all':
                    data[k] = self.transforms[k](np.array(data[k]))
        return data

    def line_mapper(self, line):
        text = line
        data = self.preprocess(text)
        return data

    def __getitem__(self, index):
        # start = time.time()
        idx = self.data_size_batch <= index
        # print('idx', idx, self.data_size_batch, index)
        file = self.files[idx][0]
        idx_in_file = index - self.data_size_batch[idx][-1]
        # print('test', file, idx_in_file)
        # read file
        # if self.current_file == file:
        #     data = self.current_file_data
        # else:
        #     data = open(file, 'r').readlines()
        #     # del self.current_file
        #     # del self.current_file_data
        #     # self.current_file = file
        #     # self.current_file_data = data
        data = self.files_data[self.files[idx][0]]
        if idx_in_file + 1 >= self.sequence_len:
            data = data[idx_in_file - self.sequence_len + 1: idx_in_file + 1]

        else:
            data = data[:idx_in_file + 1]
        # start_convert = time.time()
        processed_data = [self.line_mapper(l) for l in data]

        if len(processed_data) < self.sequence_len:
            for _ in range(self.sequence_len - len(processed_data)):
                processed_data = [self.create_empty_data(processed_data[0])] + processed_data
        current_frame = processed_data[-1]
        current_frame['Audio'] = np.hstack([p['Audio'] for p in processed_data])
        # end = time.time()
        # with open('time.txt', 'a') as f:
        #     f.write(str((end - start) * 1000))
        #     f.write('\n')
        # convert audio to spectrogram
        spectrogram = self.create_spectrogram(current_frame['Audio'])

        return torch.nan_to_num(spectrogram), np.nan_to_num(current_frame['Frame'])
        # return np.nan_to_num(current_frame['Audio']), np.nan_to_num(current_frame['Frame'])

    def create_spectrogram(self, data):
        return torchaudio.compliance.kaldi.fbank(torch.Tensor(data), htk_compat=True, sample_frequency=48000,
                                                 use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0,
                                                 frame_shift=10)

    def create_empty_data(self, data):
        new_data = {}
        for k, v in data.items():
            new_data[k] = np.zeros_like(v)
        return new_data

    def __len__(self):
        return self.data_size


class SoundDataset4(Dataset):
    def __init__(self, root_dir=None, sequence_len=None, frame_mask=None, sampling_rate=48000, num_mel_bins=None,
                 transition=False, transition_mask=None, active_mod=None):
        self.root_dir = root_dir
        self.sequence_len = sequence_len
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        frame_mask = sorted(frame_mask)
        if frame_mask is not None:
            self.frame_mask = [False] * 143
            for i in frame_mask:
                self.frame_mask[i] = True
        else:
            self.frame_mask = None
        self.size = len([f for f in os.listdir(self.root_dir) if '.pkl' in f])
        self.transition = transition
        self.transition_mask = sorted(transition_mask)
        self.calc_transition = np.zeros_like(frame_mask)
        for m in self.transition_mask:
            idx = frame_mask.index(m)
            self.calc_transition[idx] = 1
            # self.calc_transition[m] = True
        self.active_mod = active_mod

    def __getitem__(self, index):
        if index >= self.sequence_len - 1:
            start = index - self.sequence_len + 1
        else:
            start = 0

        files_data = [self.read_file(os.path.join(self.root_dir, f'{i}.pkl')) for i in range(start, index + 1)]
        current_frame_file = files_data[-1]['File']
        files_data = [f for f in files_data if f['File'] == current_frame_file]
        audio_data = [f['Audio'] for f in files_data]
        # append with empty data
        audio_data = [np.zeros_like(audio_data[0])] * (self.sequence_len - len(audio_data)) + audio_data
        audio_data = np.hstack(audio_data)
        frame_data = files_data[-1]['Frame']
        if self.frame_mask is not None:
            frame_data = np.array(frame_data)[self.frame_mask]
        down_sampling = 48000 // self.sampling_rate
        # frame transition
        if self.transition == 1 and self.transition_mask is None:
            raise Exception('Transition is set to 1 but there is no transition mask')
        if self.transition == 0 and self.transition_mask is not None:
            raise Exception('Transition is set to 0 but transition mask is not None')

        if self.transition == 1:
            if len(files_data) > 1:
                prev_frame_data = np.array(files_data[-2]['Frame'])[self.frame_mask]
            else:
                prev_frame_data = np.array(files_data[-1]['Frame'])[self.frame_mask]
                # prev_frame_data = np.zeros_like(self.frame_mask)
            frame_data = frame_data - prev_frame_data * self.calc_transition
        if self.active_mod is None:
            return self.create_spectrogram(audio_data[:, ::down_sampling]).float(), frame_data.astype(np.float32), frame_data.astype(np.float32)
        elif self.active_mod == 'sound':
            return [self.create_spectrogram(audio_data[:, ::down_sampling]).float(), frame_data.astype(np.float32)]
        elif self.active_mod == 'frame':
            return [frame_data.astype(np.float32), frame_data.astype(np.float32)]
        else:
            raise Exception('modality type not valid')

    @lru_cache(maxsize=1000)
    def read_file(self, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        return data
        # return pickle.load(open(file, 'rb'))

    def create_spectrogram(self, data):
        fbank = torchaudio.compliance.kaldi.fbank(torch.Tensor(data), htk_compat=True,
                                                  sample_frequency=self.sampling_rate,
                                                  use_energy=False, window_type='hanning',
                                                  num_mel_bins=self.num_mel_bins, dither=0.0,
                                                  frame_shift=10)
        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        return fbank

    def __len__(self):
        return self.size
