import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import random


left_ear = 234
right_ear = 454


class CustomDataset(Dataset):

    def __init__(self, directory, file_list, labels, random_rotation, keep_iris):
        self.directory = directory
        self.file_list = file_list
        self.labels = labels
        self.random_rotation = random_rotation
        self.default_inter_ear_distance = 0.14717439
        self.keep_iris = keep_iris


    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        file_name = os.path.join(self.directory, self.file_list[idx])
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(0)

        data = np.load(file_name)

        # Make the y axis normalized
        data[:,:,1] *= (1080 / 1920)

        # Translate the origin to the center of all points
        mean = data.mean(axis=(0, 1))
        data = data - mean[np.newaxis, np.newaxis, :]

        if self.random_rotation:
            # Rotate the data
            data = data.dot(R.random().as_matrix())

        # Scale to inter-ear distance
        scaling_factor = self.default_inter_ear_distance / inter_ear_distance(data[0])
        data *= scaling_factor

        data = np.float32(data)

        vid_id = self.file_list[idx].split("_$_")[0]
        vid_start = int(self.file_list[idx].split("_$_")[1][:-4])

        return torch.from_numpy(data), label, vid_id, vid_start, ""


def get_dataloaders(directory, csv_file_path, train_split=None, test_split=0.2, val_split=0.2, batch_size=32, 
                    train_workers=8, test_workers=1, val_workers=4, dual_channel=False, random_rotation=False, 
                    keep_iris=True, mesh_order_shuffle=False, blendshapes=False, single_class=None):
    
    file_list = sorted(os.listdir(directory))
    csv_data = pd.read_csv(csv_file_path)
    
    train_dataset, test_dataset, val_dataset = get_datasets_unique_chunks(csv_data, 
                                                                              file_list, 
                                                                              directory, 
                                                                              train_split, 
                                                                              test_split=test_split, 
                                                                              val_split=val_split, 
                                                                              random_rotation=random_rotation,
                                                                              keep_iris=keep_iris,
                                                                              blendshapes=blendshapes,
                                                                              single_class=single_class)


    train_dataloader, test_dataloader, val_dataloader = None, None, None
    if train_workers > 0:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                      num_workers=train_workers, persistent_workers=True)
    if test_workers > 0:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                     num_workers=test_workers, persistent_workers=True)
    if val_workers > 0:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                    num_workers=val_workers, persistent_workers=True)
    return train_dataloader, test_dataloader, val_dataloader


def get_datasets_unique_chunks(csv_data, file_list, directory, train_split=None, test_split=0.2, val_split=0.2, 
                               random_rotation=False, keep_iris=True, blendshapes=False, single_class=None):
    
    file_list = sorted(list(set([file.split("_$_")[0] for file in file_list])))
    label_list = [int(csv_data[csv_data["vid_id"] == file]["president"].values[0] == 'obama') for file in file_list]

    if single_class is not None:
        file_list = [file for file, label in zip(file_list, label_list) if label == single_class]
        label_list = [label for label in label_list if label == single_class]

    if test_split == 0.0:
        X_train = file_list
        y_train = label_list
        X_test = []
        y_test = []
    else:
        X_train, X_test, y_train, y_test = train_test_split(file_list, label_list, stratify=label_list, 
                                                            train_size=train_split,
                                                            test_size=test_split, random_state=1)
        
    if val_split == 0.0:
        X_val = []
        y_val = []
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train,
                                                          test_size=val_split / (1 - test_split), random_state=1)
    
    # Assert that no file is in more than one set
    assert(len(set(X_train).intersection(set(X_test))) == 0)
    assert(len(set(X_train).intersection(set(X_val))) == 0)
    assert(len(set(X_test).intersection(set(X_val))) == 0)
    
    print("Train set: ", len(X_train), "Test set: ", len(X_test), "Val set: ", len(X_val))
    
    file_list = os.listdir(directory)

    X_train = [file for file in tqdm(file_list) if file.split("_$_")[0] in X_train]
    X_val = [file for file in tqdm(file_list) if file.split("_$_")[0] in X_val]
    X_test = [file for file in tqdm(file_list) if file.split("_$_")[0] in X_test]

    # Assert that no chunk is in more than one set
    assert(len(set(X_train).intersection(set(X_test))) == 0)
    assert(len(set(X_train).intersection(set(X_val))) == 0)
    assert(len(set(X_test).intersection(set(X_val))) == 0)

    y_train = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["president"].values[0] == 'obama') 
            for file in tqdm(X_train)]
    y_val = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["president"].values[0] == 'obama') 
            for file in tqdm(X_val)]
    y_test = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["president"].values[0] == 'obama') 
            for file in tqdm(X_test)]
    
    # assert(len(X_train) + len(X_val) + len(X_test) == len(label_list))
    # assert(len(y_train) + len(y_val) + len(y_test) == len(label_list))

    print("Train: ", len(y_train), "Test: ", len(y_test), "Val: ", len(y_val))
    print("Train 1s: ", sum(y_train), "Train 0s: ", len(y_train) - sum(y_train))
    print("Test 1s: ", sum(y_test), "Test 0s: ", len(y_test) - sum(y_test))
    print("Val 1s: ", sum(y_val), "Val 0s: ", len(y_val) - sum(y_val))
    
    train_dataset = CustomDataset(directory, X_train, y_train, random_rotation, keep_iris)
    test_dataset = CustomDataset(directory, X_test, y_test, random_rotation, keep_iris)
    val_dataset = CustomDataset(directory, X_val, y_val, random_rotation, keep_iris)

    return train_dataset, test_dataset, val_dataset


def inter_ear_distance(data):
   return np.linalg.norm(data[left_ear] - data[right_ear])


