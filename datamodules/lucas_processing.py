import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
import random


class ModelReadyDataset(Dataset):
    """Torch Dataset for model-ready data.

    Args:
        shots (list): List of shots.
        max_length (int): Maximum length of the input sequence.

    Attributes:
        input_embeds (list): List of input embeddings.
        labels (list): List of labels.
        shot_inds (list) = List of shot inds
        shot_lens (list): List of shot lengths.
    """

    def __init__(
        self,
        shots,
        end_cutoff,
        end_cutoff_timesteps,
        machine_hyperparameters,
        max_length=2048,
    ):
        self.inputs_embeds = []
        self.labels = []

        for shot in shots:
            shot_df = shot["data"]
            o = torch.tensor(
                [shot["label"] * machine_hyperparameters[shot["machine"]]],
                dtype=torch.float32,
            )

            if end_cutoff:
                shot_end = int(len(shot_df) * (end_cutoff))
            elif end_cutoff_timesteps:
                shot_end = int(len(shot_df) - end_cutoff_timesteps)

            d = torch.tensor(shot_df[:shot_end])

            # test if the shot's length is between 25 and max_length
            if 25 <= len(d) <= max_length:
                self.inputs_embeds.append(d)
                self.labels.append(o)

    def robustly_scale(self):
        """Robustly scale the data.

        Returns:
            scaler (object): Scaler used to scale the data."""

        scaler = RobustScaler()
        combined = torch.cat(self.inputs_embeds)
        scaler.fit(combined)
        for i in range(len(self.inputs_embeds)):
            self.inputs_embeds[i] = torch.from_numpy(
                scaler.transform(self.inputs_embeds[i]).astype("float32")
            )

        return scaler

    def robustly_scale_with_another_scaler(self, scaler):
        """Robustly scale the data with another scaler.

        Args:
            scaler (object): Scaler to use to scale the data.
        """

        for i in range(len(self.inputs_embeds)):
            self.inputs_embeds[i] = torch.from_numpy(
                scaler.transform(self.inputs_embeds[i])
            )

        return

    def __len__(self):
        return len(self.inputs_embeds)

    def __getitem__(self, idx):
        """
        Returns: a tuple of (input_embeds, labels) where
            inputs_embeds (tensor): Input embeddings.
            labels (tensor): a 0/1 label for disruptions/no disruption
        """
        return (
            self.inputs_embeds[idx],
            self.labels[idx],
        )


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


def get_train_test_indices_from_Jinxiang_cases(dataset, case_number, new_machine):
    """Get train and test indices for Jinxiang's cases.

    Args:
        dataset (object): Data to split.
        case_number (int): Case number.
        new_machine (str): Name of the new machine.

    Returns:
        train_indices (list): List of indices for the training set.
        test_indices (list): List of indices for the testing set.
    """

    existing_machines = {"cmod", "d3d", "east"}
    existing_machines.remove(new_machine)
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

    test_indices = random.sample(
        new_machine_indices["non_disruptive"],
        len(new_machine_indices["non_disruptive"]) // 8,
    )

    if case_number in {1, 3, 4, 5, 7, 8, 9, 11, 12}:
        if len(new_machine_indices["disruptive"]) > 40:
            disruptive_samples = random.sample(new_machine_indices["disruptive"], 40)
            train_indices.extend(disruptive_samples[:20])
            test_indices.extend(disruptive_samples[20:])
        else:
            train_indices.extend(random.sample(new_machine_indices["disruptive"], 20))
            test_indices.extend(
                random.sample(
                    new_machine_indices["disruptive"],
                    len(new_machine_indices["disruptive"]) // 2,
                )
            )

    random.shuffle(train_indices)

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
