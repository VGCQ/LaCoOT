import numpy as np
from torch import Generator
from torch.utils.data import random_split, Dataset


class MapDataset(Dataset):
    def __init__(self, dataset, map_fn, with_target=False):
        self.dataset = dataset
        self.map = map_fn
        self.with_target = with_target

    def __getitem__(self, index):
        if self.with_target:
            return self.map(self.dataset[index][0], self.dataset[index][1])
        else:
            return self.map(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


def split_dataset(dataset, percentage, random_seed, ds_name="", train_percentage: float = 1.0):
    if ds_name == "cityscapes" and 0 < train_percentage <= 1:
        dataset_length = len(dataset)
        train_length = int(np.floor(train_percentage * dataset_length))
        empty_length = dataset_length - train_length
        partial_train_ds, _ = random_split(dataset, [train_length, empty_length],
                                           generator=Generator().manual_seed(random_seed))
        valid_length = int(np.floor(percentage * train_length))
        train_length = train_length - valid_length
        train_dataset, valid_dataset = random_split(partial_train_ds, [train_length, valid_length],
                                                    generator=Generator().manual_seed(random_seed))
        return train_dataset, valid_dataset
    else:
        # If percentage is between [0, 1) we treat it as a percentage
        if 0 <= percentage < 1:
            dataset_length = len(dataset)
            valid_length = int(np.floor(percentage * dataset_length))
            train_length = dataset_length - valid_length
            train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length],
                                                        generator=Generator().manual_seed(random_seed))

            return train_dataset, valid_dataset
        # if percentage value is greater than 1 we use it as integer,
        # and it's the number of validation samples we want
        else:
            valid_length = int(percentage)
            train_length = len(dataset) - valid_length
            train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length],
                                                        generator=Generator().manual_seed(random_seed))

            return train_dataset, valid_dataset