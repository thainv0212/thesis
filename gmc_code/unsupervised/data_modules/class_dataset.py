import os
import torch
import torchaudio
import numpy as np
import json
import pickle

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset
from gmc_code.unsupervised.data_modules.extra.mhd_dataset import MHDDataset, SoundDataset3, SoundDataset4
from sklearn.model_selection import train_test_split
from typing import Optional
from tqdm import tqdm


class ClassificationDataModule(LightningDataModule):
    def __init__(self, dataset, data_dir, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = dataset
        self.data_dir = data_dir
        self.data_config = data_config

        # Data-specific variables - fill with setup function;
        self.transform = None
        self.train_data, self.val_data, self.test_data = None, None, None

    def prepare_data(self):

        if self.dataset == "mhd":
            train_data_file = os.path.join(self.data_dir, "mhd_train.pt")
            test_data_file = os.path.join(self.data_dir, "mhd_test.pt")

            if not os.path.exists(train_data_file) or not os.path.exists(test_data_file):
                raise RuntimeError('MHD Dataset not found. Please generate dataset and place it in the data folder.')
        else:
            raise ValueError(
                "[Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")

    def setup(self, stage=None):

        # Setup Dataset:
        if self.dataset == "mhd":

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                mhd_full = MHDDataset(self.data_dir, train=True)
                ## use only small part
                mhd_full, _ = random_split(mhd_full,
                                           [int(0.99 * len(mhd_full)),
                                            len(mhd_full) - int(0.99 * len(mhd_full))])
                self.train_data, self.val_data = random_split(mhd_full,
                                                              [int(0.9 * len(mhd_full)),
                                                               len(mhd_full) - int(0.9 * len(mhd_full))])
                self.dims = tuple(self.train_data[0][0].shape)

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.test_data = MHDDataset(self.data_dir, train=False)
                self.dims = tuple(self.test_data[0][0].shape)

        else:
            raise ValueError(
                "[Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.data_config["batch_size"],
            shuffle=True,
            num_workers=self.data_config["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config["num_workers"],
        )


class DCADataModule(LightningDataModule):
    def __init__(self, dataset, data_dir, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = dataset
        self.data_dir = data_dir
        self.data_config = data_config

        # Data-specific variables - fill with setup function;
        self.transform = None
        self.train_data, self.val_data, self.test_data = None, None, None
        self.test_sampler = None
        self.dca_partial_eval_indices = None

    def set_dca_eval_sample_indices(self):
        if self.dca_partial_eval_indices is None:
            self.dca_partial_eval_indices = np.random.choice(
                list(range(len(self.test_data))),
                self.data_config["n_dca_samples"],
                replace=False,
            )

    def prepare_data(self):
        # download
        if self.dataset == "mhd":
            train_data_file = os.path.join(self.data_dir, "mhd_train.pt")
            test_data_file = os.path.join(self.data_dir, "mhd_test.pt")

            if not os.path.exists(train_data_file) or not os.path.exists(test_data_file):
                raise RuntimeError('MHD Dataset not found. Please generate dataset and place it in the data folder.')

        else:
            raise ValueError(
                "[DCA Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")

    def setup(self, stage=None):

        # Setup Dataset:
        if self.dataset == "mhd":

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                mhd_full = MHDDataset(self.data_dir, train=True)
                self.train_data, self.val_data = random_split(mhd_full,
                                                              [int(0.9 * len(mhd_full)),
                                                               len(mhd_full) - int(0.9 * len(mhd_full))])
                self.dims = tuple(self.train_data[0][0].shape)

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.test_data = MHDDataset(self.data_dir, train=False)
                self.dims = tuple(self.test_data[0][0].shape)

                self.set_dca_eval_sample_indices()
                self.partial_test_sampler = torch.utils.data.SubsetRandomSampler(
                    self.dca_partial_eval_indices
                )

        else:
            raise ValueError(
                "[DCA Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.data_config["batch_size"],
            shuffle=True,
            num_workers=self.data_config["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config["num_workers"],
            sampler=self.partial_test_sampler,
            drop_last=False,
        )


class ICEDataModule(LightningDataModule):
    def __init__(self, root_dir=None, transforms=None, batch_size=32, frame_col=None):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_full = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.data_predict = None
        self.batch_size = batch_size
        self.frame_col = frame_col
        self.transforms = transforms

    def prepare_data(self) -> None:
        # self.data_full = SoundDataset(root_dir=self.root_dir, transforms=self.transforms)
        pass

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # self.data_full = SoundDataset3(root_dir=self.root_dir, transforms=self.transforms, frame_col=self.frame_col)
            # mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            # self.data_train, self.data_val = self.data_full.train_test_split()
            files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir)]
            files_train, files_val = train_test_split(files, test_size=0.2)
            self.data_train = SoundDataset3(files=files_train, transforms=self.transforms, frame_col=self.frame_col)
            self.data_val = SoundDataset3(files=files_val, transforms=self.transforms, frame_col=self.frame_col)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = SoundDataset3(root_dir=self.root_dir, transforms=self.transforms, frame_col=self.frame_col)

        if stage == "predict" or stage is None:
            self.data_predict = SoundDataset3(root_dir=self.root_dir, transforms=self.transforms,
                                              frame_col=self.frame_col)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size)


class ICEDataModule2(LightningDataModule):
    def __init__(self, train_dir=None, test_dir=None, batch_size=32, frame_mask=None, sampling_rate=None,
                 data_config=None, sound_length=None, num_mel_bins=None, transition=False, transition_mask=None,
                 active_mod=None):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.data_full = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.data_predict = None
        self.batch_size = batch_size
        self.frame_mask = frame_mask
        self.sampling_rate = sampling_rate
        self.data_config = data_config
        self.sound_length = sound_length
        self.num_mel_bins = num_mel_bins
        self.transition = transition
        self.transition_mask = transition_mask
        self.active_mod = active_mod
        print(f'active modality: {active_mod}')

    def prepare_data(self) -> None:
        # self.data_full = SoundDataset(root_dir=self.root_dir, transforms=self.transforms)
        pass

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # self.data_full = SoundDataset3(root_dir=self.root_dir, transforms=self.transforms, frame_col=self.frame_col)
            # mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            # self.data_train, self.data_val = self.data_full.train_test_split()
            # files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir)]
            # files_train, files_val = train_test_split(files, test_size=0.2)
            # self.data_train = SoundDataset3(files=files_train, transforms=self.transforms, frame_col=self.frame_col)
            # self.data_val = SoundDataset3(files=files_val, transforms=self.transforms, frame_col=self.frame_col)
            sequence_length = self.sound_length // 800
            self.data_train = SoundDataset4(root_dir=self.train_dir, frame_mask=self.frame_mask,
                                            sampling_rate=self.sampling_rate, sequence_len=sequence_length,
                                            num_mel_bins=self.num_mel_bins, transition=self.transition,
                                            transition_mask=self.transition_mask, active_mod=self.active_mod)
            self.data_val = SoundDataset4(root_dir=self.test_dir, frame_mask=self.frame_mask,
                                          sampling_rate=self.sampling_rate, sequence_len=sequence_length,
                                          num_mel_bins=self.num_mel_bins, transition=self.transition,
                                          transition_mask=self.transition_mask, active_mod=self.active_mod)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = SoundDataset4(root_dir=self.test_dir, frame_mask=self.frame_mask,
                                           sampling_rate=self.sampling_rate, sequence_len=sequence_length,
                                           num_mel_bins=self.num_mel_bins, transition=self.transition,
                                           transition_mask=self.transition_mask, active_mod=self.active_mod)

        if stage == "predict" or stage is None:
            self.data_predict = SoundDataset4(root_dir=self.test_dir, frame_mask=self.frame_mask,
                                              sampling_rate=self.sampling_rate, sequence_len=sequence_length,
                                              num_mel_bins=self.num_mel_bins, transition=self.transition,
                                              transition_mask=self.transition_mask, active_mod=self.active_mod)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.data_config['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.data_config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.data_config['num_workers'])

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size, num_workers=self.data_config['num_workers'])
