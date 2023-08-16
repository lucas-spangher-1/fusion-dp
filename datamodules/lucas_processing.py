import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
import random
from typing import List
import math


class ModelReadyDataset(Dataset):
    """Torch Dataset for model-ready data.

    Args:
        shots (list): List of shots.
        inds (list): List of shot indices for reference.
        end_cutoff (float): Fraction of the shot to use as the end.
        end_cutoff_timesteps (int): Number of timesteps to cut off the end of the shot.
        max_length (int): Maximum length of the input sequence.

    Attributes:
        xs (list): List of input embeddings.
        ys (list): List of labels.
        metas (list) = List of shot metadata like machine, index, etc
    """

    def __init__(
        self,
        shots: List,
        inds: List,
        end_cutoff,
        end_cutoff_timesteps,
        machine_hyperparameters,
        taus,
        max_length=2048,
        len_aug: bool = False,
        seed: int = 42,
        len_aug_args: dict = {},
    ):
        self.len_aug = len_aug
        self.len_aug_args = len_aug_args
        self.rand = random.Random(seed)
        self.taus = taus

        self.xs = []
        self.ys = []
        self.metas = []
        self.machines = []

        for shot, ind in zip(shots, inds):
            shot_df = shot["data"]
            o = torch.tensor(
                [shot["label"] * machine_hyperparameters[shot["machine"]]],
                dtype=torch.float32,
            )

            shot_end = 0
            if end_cutoff:
                shot_end = int(len(shot_df) * (end_cutoff))
            elif end_cutoff_timesteps:
                shot_end = int(len(shot_df) - end_cutoff_timesteps)
            else:
                raise Exception(
                    "Must provide either end_cutoff or end_cutoff_timesteps"
                )

            d = torch.tensor(shot_df[:shot_end], dtype=torch.float32)

            # test if the shot's length is between 15 and max_length
            if 15 <= len(d) <= max_length:
                self.xs.append(d)
                self.ys.append(o)
                self.metas.append(
                    {
                        "ind": ind,
                        "shot_len": shot_end,
                        "machine": shot["machine"],
                    }
                )

    def robustly_scale(self):
        """Robustly scale the data.

        Returns:
            scaler (object): Scaler used to scale the data."""

        scaler = RobustScaler()
        combined = torch.cat(self.xs)
        scaler.fit(combined)
        for i in range(len(self.xs)):
            self.xs[i] = torch.from_numpy(
                scaler.transform(self.xs[i]).astype("float32")
            )

        return scaler

    def robustly_scale_with_another_scaler(self, scaler):
        """Robustly scale the data with another scaler.

        Args:
            scaler (object): Scaler to use to scale the data.
        """

        for i in range(len(self.xs)):
            self.xs[i] = torch.from_numpy(
                scaler.transform(self.xs[i]).astype("float32")
            )

        return

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        """
        Returns: a tuple of (input_embeds, labels, len) where
            inputs_embeds (tensor): Input embeddings.
            labels (tensor): a 0/1 label for disruptions/no disruption
            len (int): Length of the shot.
        """
        x, y, length = self.xs[idx], self.ys[idx], self.metas[idx]["shot_len"]
        m = self.metas[idx]["machine"]
        tau = self.taus[m]

        if self.len_aug:
            x, y, length = length_augmentation(
                x, y, length, tau, self.rand, **self.len_aug_args
            )

        assert x.shape[0] == length
        return x, y, length


def get_train_test_indices_from_Jinxiang_cases(
    dataset, case_number, new_machine, seed, case8_percentage=0.125
):
    """Get train and test indices for Jinxiang's cases.

    Args:
        dataset (object): Data to split.
        case_number (int): Case number.
        new_machine (str): Name of the new machine.

    Returns:
        train_indices (list): List of indices for the training set.
        test_indices (list): List of indices for the testing set.
    """

    rand = random.Random(seed)

    existing_machines = {"cmod", "d3d", "east"}
    existing_machines.remove(new_machine)

    assert case_number == 8, "Case number not supported"
    if case_number == 8:
        # Take case8_percentage of the data from each machine for test,
        # this ensures each machine has the same percentage of data in the test set
        by_machine = {"cmod": set(), "d3d": set(), "east": set()}
        test_indices = set()
        for key, value in dataset.items():
            by_machine[value["machine"]].add(key)
        for _machine, inds in by_machine.items():
            test_indices.update(
                rand.sample(sorted(inds), int(case8_percentage * len(inds)))
            )

        train_indices = list(set(dataset.keys()) - test_indices)
        rand.shuffle(train_indices)
        return train_indices, list(test_indices)
    return ([], [])


def length_augmentation(
    x,
    y,
    length,
    tau,
    rand: random.Random,
    tiny_clip_max_len=30,
    tiny_clip_prob=0.05,
    disrupt_trim_max=10,
    disrupt_trim_prob=0.2,
    nondisr_cut_min=15,
    nondisr_cut_prob=0.3,
    tau_trim_prob=0.2,
    tau_trim_max=10,
):
    # TODO: actually do a logical or on all these cases
    """Perform length augmentation clipping.

    Args:
        x (torch.Tensor): the input x tensor
        y (torch.Tensor): the torch tensor with the y input
        length (int): the length of x
        rand (random.Random): the random sampler to draw from
        tiny_clip_max_len (int, optional): The maximum length to trim a sequence to when
            doing tiny clipping. Defaults to 30.
        tiny_clip_prob (float, optional): The probability of doing tiny clipping.
            Defaults to 0.05.
        disrupt_trim_max (int, optional): The maximum amount to remove from the end of
            a disruption when doing trimming. Defaults to 10.
        disrupt_trim_prob (float, optional): The probability of doing disruption
            trimming. Defaults to 0.2.
        nondisr_cut_min (int, optional): The minimum size to cut non-disruptions to.
            Defaults to 15.
        nondisr_cut_prob (float, optional): The probability of doing non-disruption
            trimming. Defaults to 0.3.
        tau_trim_prob (float, optional): the probability we do tau trimming
        tau_trim_max (int, optional): the maximum we cut from the end of the seq
            when doing tau trimming

    Returns:
        (x, y, len): the x, y, len to use
    """
    new_len = None

    if rand.random() < tiny_clip_prob:
        # sample len in [1, tiny_clip_max_len]
        new_len = math.ceil(rand.random() * tiny_clip_max_len)
        new_len = min(new_len, length)
        return x[:new_len], torch.tensor(0), new_len

    elif y == 1 and rand.random() < disrupt_trim_prob:
        # sample len in [len-disrupt_trim_max, len]
        new_len = length - math.floor(rand.random() * disrupt_trim_max)
        new_len = min(new_len, length)
        return x[:new_len], y, new_len

    elif y == 1 and rand.random() < tau_trim_prob:
        # sample len in [len-tau_trim_max, len]
        new_len = length - math.floor(rand.random() * tau_trim_max)
        new_len = min(new_len, length)
        if new_len < length - tau:
            y = 0
        return x[:new_len], y, new_len

    elif y == 0 and rand.random() < nondisr_cut_prob:
        # sample len in [nondisr_cut_min, len]
        new_len = nondisr_cut_min + math.ceil(
            (length - nondisr_cut_min) * rand.random()
        )
        new_len = min(new_len, length)
        return x[:new_len], y, new_len

    else:
        return x, y, length
