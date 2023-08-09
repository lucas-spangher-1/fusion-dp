import math
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from . import lucas_processing
import pickle
from torch.utils.data import random_split, DataLoader
import torch
from torch import Generator
import requests
import gzip
from tqdm import tqdm
import os


def collate_fn(batch):
    inputs, labels, lengths = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True).transpose(1, 2)
    labels = torch.tensor(labels)
    return inputs, labels, lengths


class StreamingProgressResponse:
    def __init__(self, response, progress):
        self.response = response
        self.progress = progress

    def read(self, size=-1):
        chunk = self.response.raw.read(size)
        self.progress.update(len(chunk))
        return chunk


class LucasDataModule(pl.LightningDataModule):
    """
    DataModule for Lucas' fusion dataset.
    """

    DATA_FILENAME = "lucas_data_f32.pickle"

    # TODO: add sample rate here
    def __init__(
        self,
        data_dir: str,
        pin_memory: bool,
        data_type: str = "default",
        end_cutoff: int = None,
        end_cutoff_timesteps: int = None,
        new_machine: str = "cmod",
        case_number: int = 8,
        machine_hyperparameters: dict = {"cmod": 1.0, "d3d": 1.0, "east": 1.0},
        batch_size: int = 32,
        test_batch_size: int = 32,
        generator: Optional[Generator] = None,
        val_percent: float = 0.1,
        num_workers: int = 1,
        augment: bool = False,
        debug: bool = False,
        seed: int = 42,
        len_aug_args: dict = {},
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.pin_memory = pin_memory
        self.end_cutoff = end_cutoff
        self.end_cutoff_timesteps = end_cutoff_timesteps
        self.new_machine = new_machine
        self.case_number = case_number
        self.machine_hyperparameters = machine_hyperparameters
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.generator = generator
        self.val_percent = val_percent
        self.num_workers = num_workers
        self.augment = augment
        self.len_aug_args = len_aug_args
        self.debug = debug
        self.seed = seed

        if data_type != "default" and data_type != "sequence":
            raise ValueError(f"data_type {data_type} not supported.")

        self.data_dim = 1
        self.data_type = "sequence"

        self.input_channels = 13
        self.output_channels = 1

        if end_cutoff is None and end_cutoff_timesteps is None:
            raise ValueError("Must specify either end_cutoff or end_cutoff_timesteps.")

        if end_cutoff is not None and end_cutoff_timesteps is not None:
            raise ValueError(
                "Must specify either end_cutoff or end_cutoff_timesteps, not both."
            )

    def prepare_data(self):
        """
        TODO: Download this in the future
        """

        URL = "https://pub-651766aedb444e189ec2533015f228de.r2.dev/lucas_data_f32.pickle.gzip"

        if os.path.exists(os.path.join(self.data_dir, self.DATA_FILENAME)):
            return

        os.makedirs(self.data_dir, exist_ok=True)

        resp = requests.get(URL, stream=True)
        progress = tqdm(
            desc="Downloading lucas_data_f32.pickle",
            total=int(resp.headers["Content-Length"]),
            unit="B",
            unit_scale=True,
        )
        resp = StreamingProgressResponse(resp, progress)
        f = gzip.GzipFile(fileobj=resp)
        data = pickle.load(f)
        progress.close()
        # Save to self.DATA_FILENAME
        with open(os.path.join(self.data_dir, self.DATA_FILENAME), "wb") as f:
            pickle.dump(data, f)

    def setup(self, stage=None):
        # Load data from file
        f = open(os.path.join(self.data_dir, self.DATA_FILENAME), "rb")
        data = pickle.load(f)

        (
            train_inds,
            test_inds,
        ) = lucas_processing.get_train_test_indices_from_Jinxiang_cases(
            dataset=data,
            case_number=self.case_number,
            new_machine=self.new_machine,
            seed=self.seed,
        )

        if self.debug:
            train_inds = train_inds[:80]
            test_inds = test_inds[:20]

        n_val = int(round(len(train_inds) * self.val_percent))

        val_shots = [data[i] for i in train_inds[:n_val]]
        train_shots = [data[i] for i in train_inds[n_val:]]
        test_shots = [data[i] for i in test_inds]

        self.train_dataset = lucas_processing.ModelReadyDataset(
            shots=train_shots,
            inds=train_inds,
            machine_hyperparameters=self.machine_hyperparameters,
            end_cutoff=self.end_cutoff,
            end_cutoff_timesteps=self.end_cutoff_timesteps,
            len_aug=self.augment,
            len_aug_args=self.len_aug_args,
        )
        self.val_dataset = lucas_processing.ModelReadyDataset(
            shots=val_shots,
            inds=train_inds,
            machine_hyperparameters=self.machine_hyperparameters,
            end_cutoff=self.end_cutoff,
            end_cutoff_timesteps=self.end_cutoff_timesteps,
        )
        self.test_dataset = lucas_processing.ModelReadyDataset(
            shots=test_shots,
            inds=test_inds,
            machine_hyperparameters=self.machine_hyperparameters,
            end_cutoff=self.end_cutoff,
            end_cutoff_timesteps=self.end_cutoff_timesteps,
        )
        # TODO: hardcode the scaler values in the future so we don't need to load
        # both train/test always
        scaler = self.train_dataset.robustly_scale()

        self.test_dataset.robustly_scale_with_another_scaler(scaler)
        self.val_dataset.robustly_scale_with_another_scaler(scaler)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dl = DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return dl

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.test_dataset,
            self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(
            self.val_dataset,
            self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return dl

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
