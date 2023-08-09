import torch
from torch.nn.utils.rnn import pad_sequence
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
        max_length=2048,
        len_aug: bool = False,
        seed: int = 42,
        len_aug_args: dict = {},
    ):
        self.len_aug = len_aug
        self.len_aug_args = len_aug_args
        self.rand = random.Random(seed)

        self.xs = []
        self.ys = []
        self.metas = []

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

            d = torch.tensor(shot_df[:shot_end])

            # test if the shot's length is between 25 and max_length
            if 25 <= len(d) <= max_length:
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
        # for length augmentation, we clip up to
        x, y, len = self.xs[idx], self.ys[idx], self.metas[idx]["shot_len"]

        if self.len_aug:
            x, y, len = length_augmentation(x, y, len, self.rand, **self.len_aug_args)

        assert x.shape[0] == len
        return x, y, len


def collate_fn_seq_to_label(dataset):
    """
    Takes in an instance of Torch Dataset.
    Returns:
     * input_embeds: tensor, size: Batch x (padded) seq_length x embedding_dim
     * label_ids: tensor, size: Batch x 1 x 1
    """

    output = {}

    output["inputs_embeds"] = pad_sequence(
        [df["inputs_embeds"].to(dtype=torch.float16) for df in dataset],
        padding_value=-10,
        batch_first=True,
    )
    output["labels"] = torch.tensor([df["labels"].to(torch.long) for df in dataset])
    output["attention_mask"] = (output["inputs_embeds"][:, :, 1] != -10).to(torch.long)

    return output


def collate_fn_seq_to_seq(dataset):
    """
    Takes in an instance of Torch Dataset and collates sequences for inputs and outputs.
    Returns:
     * input_embeds: tensor, size: Batch x (padded) seq_length x embedding_dim
     * label_ids: tensor, size: Batch x (padded) seq_length x 1
    """

    output = {}

    output["inputs_embeds"] = pad_sequence(
        [df["inputs_embeds"].to(dtype=torch.float16) for df in dataset],
        padding_value=-10,
        batch_first=True,
    )

    output["labels"] = pad_sequence(
        [df["labels"].to(dtype=torch.long) for df in dataset],
        padding_value=-10,
        batch_first=True,
    )

    output["attention_mask"] = (output["inputs_embeds"][:, :, 1] != -10).to(torch.long)

    return output


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

    # TODO: not case 8...

    train_indices = []

    new_machine_indices = {"non_disruptive": [], "disruptive": []}

    for key, value in dataset.items():
        if value["machine"] == new_machine:
            if value["label"] == 0:
                new_machine_indices["non_disruptive"].append(key)
            else:
                new_machine_indices["disruptive"].append(key)

        elif value["machine"] in existing_machines:
            if case_number in {4, 5, 6, 7, 8, 9, 10, 11, 12} and value["label"] == 0:
                train_indices.append(key)
            if case_number in {1, 2, 3, 5, 6, 7, 8, 10, 12} and value["label"] == 1:
                train_indices.append(key)
            if case_number == 11 and value["label"] == 1:
                train_indices.append(key)

    if case_number == 3:
        half_size = len(train_indices) // 2
        train_indices = train_indices[:half_size]
    if case_number == 11:
        fifth_size = len(train_indices) // 5
        train_indices = train_indices[:fifth_size]

    if case_number in {1, 2, 4, 5, 6, 8, 9, 10, 11, 12}:
        train_indices.extend(new_machine_indices["non_disruptive"])

    test_indices = rand.sample(
        new_machine_indices["non_disruptive"],
        len(new_machine_indices["non_disruptive"]) // 8,
    )

    if case_number in {1, 3, 4, 5, 7, 8, 9, 11, 12}:
        if len(new_machine_indices["disruptive"]) > 40:
            disruptive_samples = rand.sample(new_machine_indices["disruptive"], 40)
            train_indices.extend(disruptive_samples[:20])
            test_indices.extend(disruptive_samples[20:])
        else:
            train_indices.extend(rand.sample(new_machine_indices["disruptive"], 20))
            test_indices.extend(
                random.sample(
                    new_machine_indices["disruptive"],
                    len(new_machine_indices["disruptive"]) // 2,
                )
            )

    rand.shuffle(train_indices)

    # Create test set by sampling 20% of the new machine's shots

    # Remove test indices from training set if they were added earlier
    train_indices = [index for index in train_indices if index not in test_indices]

    return train_indices, test_indices


def get_class_weights(train_dataset):
    """Get class weights for the training set.

    Args:
        train_dataset (object): Training set.

    Returns:
        class_weights (list): List of class weights.
    """
    class_counts = {}

    for i in range(len(train_dataset)):
        df = train_dataset[i]
        label = int(df["labels"][0])
        if label in class_counts.keys():
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    class_weights = [
        class_counts[key] / sum(class_counts.values()) for key in class_counts.keys()
    ]

    print("class weights: ")
    print(class_weights)

    return class_weights


def length_augmentation(
    x,
    y,
    len,
    rand: random.Random,
    tiny_clip_max_len=30,
    tiny_clip_prob=0.05,
    disrupt_trim_max=10,
    disrupt_trim_prob=0.2,
    nondisr_cut_min=15,
    nondisr_cut_prob=0.3,
):
    """Perform length augmentation clipping.

    Args:
        x (torch.Tensor): the input x tensor
        y (torch.Tensor): the torch tensor with the y input
        len (int): the length of x
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

    Returns:
        _type_: _description_
    """
    if rand.random() < tiny_clip_prob:
        # sample len in [1, tiny_clip_max_len]
        new_len = math.ceil(rand.random() * tiny_clip_max_len)
        return x[:new_len], torch.tensor(0), new_len
    elif y == 1 and rand.random() < disrupt_trim_prob:
        # sample len in [len-disrupt_trim_max, len]
        new_len = len - math.floor(rand.random() * disrupt_trim_max)
        return x[:new_len], y, new_len
    elif y == 0 and rand.random() < nondisr_cut_prob:
        # sample len in [nondisr_cut_min, len]
        new_len = nondisr_cut_min + math.ceil((len - nondisr_cut_min) * rand.random())
        return x[:new_len], y, new_len
    else:
        return x, y, len


def get_class_weights_seq_to_seq(train_dataset):
    """Get class weights for the training set.

    Args:
        train_dataset (object): Training set.

    Returns:
        class_weights (list): List of class weights.
    """
    class_counts = {0: 0, 1: 0}

    for i in range(len(train_dataset)):
        df = train_dataset[i]
        ones = torch.sum(df["labels"])
        zeros = len(df["labels"]) - ones
        class_counts[0] += zeros
        class_counts[1] += ones
    class_weights = [
        class_counts[key] / sum(class_counts.values()) for key in class_counts.keys()
    ]

    print("class weights: ")
    print(class_weights)

    return class_weights
