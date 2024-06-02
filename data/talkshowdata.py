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
iris_keypoints = [469, 470, 471, 472, 474, 475, 476, 477]
# iris_keypoints = [i for i in range(1, 478) if i != left_ear and i != right_ear]


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

        if not self.keep_iris:
            data = np.delete(data, iris_keypoints, axis=1)

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
        vid_start = int(self.file_list[idx].split("_$_")[2][:-4])
        person_id = self.file_list[idx].split("_$_")[1]

        return torch.from_numpy(data), label, vid_id, vid_start, person_id
    

class CustomDatasetDualChannel(Dataset):

    def __init__(self, directory, file_list_left, file_list_right, labels, random_rotation, keep_iris, mesh_order_shuffle):
        self.directory = directory
        self.file_list_left = file_list_left
        self.file_list_right = file_list_right
        self.labels = labels
        self.random_rotation = random_rotation
        self.default_inter_ear_distance = 0.14717439
        self.rotations_left = {}
        self.rotations_right = {}
        self.keep_iris = keep_iris
        self.mesh_order_shuffle = mesh_order_shuffle


    def __len__(self):
        return len(self.file_list_left)


    def __getitem__(self, idx):
        file_name_left = os.path.join(self.directory, self.file_list_left[idx])
        file_name_right = os.path.join(self.directory, self.file_list_right[idx])

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(0)

        data_left = np.load(file_name_left)
        data_right = np.load(file_name_right)

        # Make the y axis normalized
        data_left[:,:,1] *= (1080 / 1920)
        data_right[:,:,1] *= (1080 / 1920)

        if not self.keep_iris:
            data_left = np.delete(data_left, iris_keypoints, axis=1)
            data_right = np.delete(data_right, iris_keypoints, axis=1)
        
        # Translate the origin to the center of all points
        mean_left = data_left.mean(axis=(0, 1))
        data_left = data_left - mean_left[np.newaxis, np.newaxis, :]
        mean_right = data_right.mean(axis=(0, 1))
        data_right = data_right - mean_right[np.newaxis, np.newaxis, :]

        if self.random_rotation:
            # Rotate the data
            # rotation_left = R.random(random_state=idx).as_matrix()
            # rotation_right = R.random(random_state=idx + 1000000).as_matrix()
            rotation_left = R.random().as_matrix()
            rotation_right = R.random().as_matrix()
            data_left = data_left.dot(rotation_left)
            data_right = data_right.dot(rotation_right)

        # Scale to inter-ear distance
        scaling_factor_left = self.default_inter_ear_distance / inter_ear_distance(data_left[0])
        data_left *= scaling_factor_left
        scaling_factor_right = self.default_inter_ear_distance / inter_ear_distance(data_right[0])
        data_right *= scaling_factor_right

        data_left = np.float32(data_left)
        data_right = np.float32(data_right)

        vid_id = self.file_list_left[idx].split("_$_")[0]
        timestamp = int(self.file_list_left[idx].split("_$_")[2][:-4])

        if self.mesh_order_shuffle and bool(random.getrandbits(1)):
            data_left, data_right = data_right, data_left

        return torch.from_numpy(data_left), torch.from_numpy(data_right), label, vid_id, timestamp


class CustomDatasetBlendshapes(Dataset):

    def __init__(self, directory, file_list, labels):
        self.directory = directory
        self.file_list = file_list
        self.labels = labels


    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        file_name = os.path.join(self.directory, self.file_list[idx])

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(0)

        data = np.load(file_name)
        data = np.float32(data).squeeze()
        data -= data.mean(axis=1)[:, np.newaxis]

        # Remove iris features
        data = np.delete(data, slice(11, 23), axis=1)

        return torch.from_numpy(data), label, self.file_list[idx]


def get_dataloaders(directory, csv_file_path, train_split=None, test_split=0.2, val_split=0.2, batch_size=32, 
                    train_workers=8, test_workers=1, val_workers=4, dual_channel=False, random_rotation=False, 
                    keep_iris=True, mesh_order_shuffle=False, blendshapes=False, single_class=None, person_id=None):
    
    file_list = sorted(os.listdir(directory))
    csv_data = pd.read_csv(csv_file_path)
    
    if not dual_channel:
        train_dataset, test_dataset, val_dataset = get_datasets_unique_chunks(csv_data, 
                                                                              file_list, 
                                                                              directory, 
                                                                              train_split, 
                                                                              test_split=test_split, 
                                                                              val_split=val_split, 
                                                                              random_rotation=random_rotation,
                                                                              keep_iris=keep_iris,
                                                                              blendshapes=blendshapes,
                                                                              single_class=single_class,
                                                                              person_id=person_id)
    elif dual_channel:
        train_dataset, test_dataset, val_dataset = get_dataasets_dual_channel(csv_data, 
                                                                              file_list, 
                                                                              directory,
                                                                              train_split, 
                                                                              test_split=test_split, 
                                                                              val_split=val_split, 
                                                                              random_rotation=random_rotation,
                                                                              keep_iris=keep_iris,
                                                                              mesh_order_shuffle=mesh_order_shuffle,
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
                               random_rotation=False, keep_iris=True, blendshapes=False, single_class=None,
                               person_id=None):
    if person_id is not None:
        file_list = [file for file in file_list if file.split("_$_")[1] == person_id]

    file_list = sorted(list(set([file.split("_$_")[0] for file in file_list])))
    label_list = [int(csv_data[csv_data["vid_id"] == file]["view"].values[0] == 'off') for file in file_list]

    if single_class is not None:
        file_list = [file for file, label in zip(file_list, label_list) if label == single_class]
        label_list = [label for label in label_list if label == single_class]

    if test_split == 0.0:
        X_train = file_list
        y_train = label_list
        X_test = []
        y_test = []
    else:
        X_train, X_test, y_train, y_test = train_test_split(file_list, label_list, stratify=label_list, train_size=train_split,
                                                            test_size=test_split, random_state=3)
        
    if val_split == 0.0:
        X_val = []
        y_val = []
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train,
                                                          test_size=val_split / (1 - test_split), random_state=3)
    
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

    y_train = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["view"].values[0] == 'off') 
            for file in tqdm(X_train)]
    y_val = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["view"].values[0] == 'off') 
            for file in tqdm(X_val)]
    y_test = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["view"].values[0] == 'off') 
            for file in tqdm(X_test)]
    
    # assert(len(X_train) + len(X_val) + len(X_test) == len(label_list))
    # assert(len(y_train) + len(y_val) + len(y_test) == len(label_list))

    print("Train: ", len(y_train), "Test: ", len(y_test), "Val: ", len(y_val))
    print("Train 1s: ", sum(y_train), "Train 0s: ", len(y_train) - sum(y_train))
    print("Test 1s: ", sum(y_test), "Test 0s: ", len(y_test) - sum(y_test))
    print("Val 1s: ", sum(y_val), "Val 0s: ", len(y_val) - sum(y_val))
    
    if not blendshapes:
        train_dataset = CustomDataset(directory, X_train, y_train, random_rotation, keep_iris)
        test_dataset = CustomDataset(directory, X_test, y_test, random_rotation, keep_iris)
        val_dataset = CustomDataset(directory, X_val, y_val, random_rotation, keep_iris)
    elif blendshapes:
        train_dataset = CustomDatasetBlendshapes(directory, X_train, y_train)
        test_dataset = CustomDatasetBlendshapes(directory, X_test, y_test)
        val_dataset = CustomDatasetBlendshapes(directory, X_val, y_val)

    return train_dataset, test_dataset, val_dataset


def get_dataasets_dual_channel(csv_data, file_list, directory, train_split=None, test_split=0.2, val_split=0.2, 
                               random_rotation=False, keep_iris=True, mesh_order_shuffle=False, single_class=None):
    file_list_uniq = sorted(list(set([file.split("_$_")[0] for file in file_list])))
    label_list = [int(csv_data[csv_data["vid_id"] == file]["view"].values[0] == 'off') for file in file_list_uniq]

    if single_class is not None:
        file_list_uniq = [file for file, label in zip(file_list_uniq, label_list) if label == single_class]
        label_list = [label for label in label_list if label == single_class]

    X_train, X_test, y_train, y_test = train_test_split(file_list_uniq, label_list, stratify=label_list, train_size=train_split,
                                                        test_size=test_split, random_state=3)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train,
                                                    test_size=val_split / (1 - test_split), random_state=3)

    X_train_left = sorted([file for file in tqdm(file_list) if file.split("_$_")[0] in X_train and file.split("_$_")[1] == "left"])
    X_train_right = sorted([file for file in tqdm(file_list) if file.split("_$_")[0] in X_train and file.split("_$_")[1] == "right"])
    X_val_left = sorted([file for file in tqdm(file_list) if file.split("_$_")[0] in X_val and file.split("_$_")[1] == "left"])
    X_val_right = sorted([file for file in tqdm(file_list) if file.split("_$_")[0] in X_val and file.split("_$_")[1] == "right"])
    X_test_left = sorted([file for file in tqdm(file_list) if file.split("_$_")[0] in X_test and file.split("_$_")[1] == "left"])
    X_test_right = sorted([file for file in tqdm(file_list) if file.split("_$_")[0] in X_test and file.split("_$_")[1] == "right"])

    y_train = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["view"].values[0] == 'off') 
            for file in tqdm(X_train_left)]
    y_val = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["view"].values[0] == 'off') 
            for file in tqdm(X_val_left)]
    y_test = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["view"].values[0] == 'off') 
            for file in tqdm(X_test_left)]
    
    # Count number of 1s and 0s in each set
    print("Train: ", len(X_train_left), "Test: ", len(X_test_left), "Val: ", len(X_val_left))
    print("Train 1s: ", sum(y_train), "Train 0s: ", len(y_train) - sum(y_train))
    print("Test 1s: ", sum(y_test), "Test 0s: ", len(y_test) - sum(y_test))
    print("Val 1s: ", sum(y_val), "Val 0s: ", len(y_val) - sum(y_val))

    train_dataset = CustomDatasetDualChannel(directory, X_train_left, X_train_right, y_train, random_rotation, keep_iris, mesh_order_shuffle)
    test_dataset = CustomDatasetDualChannel(directory, X_test_left, X_test_right, y_test, random_rotation, keep_iris, mesh_order_shuffle)
    val_dataset = CustomDatasetDualChannel(directory, X_val_left, X_val_right, y_val, random_rotation, keep_iris, mesh_order_shuffle)
    return train_dataset, test_dataset, val_dataset


def inter_ear_distance(data):
   return np.linalg.norm(data[left_ear] - data[right_ear])


