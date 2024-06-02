import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm


class CustomDataset(Dataset):

    def __init__(self, directory, file_list, labels, random_rotation):
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
        data = np.float32(data)

        return torch.from_numpy(data), label
    

def get_dataloaders(directory, csv_file_path, test_split=0.2, val_split=0.2, batch_size=32, train_workers=8, 
                    test_workers=1, val_workers=4, unique_vids=False, dual_channel=False, random_rotation=False):
    
    file_list = os.listdir(directory)
    csv_data = pd.read_csv(csv_file_path)
    
    train_dataset, test_dataset, val_dataset = get_datasets_unique_chunks(csv_data, file_list, directory, 
                                                                          test_split=test_split, 
                                                                          val_split=val_split, 
                                                                          random_rotation=random_rotation)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=test_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=val_workers)

    return train_dataloader, test_dataloader, val_dataloader


def get_datasets_unique_chunks(csv_data, file_list, directory, test_split=0.2, val_split=0.2, 
                               random_rotation=False):
    file_list = list(set([file.split("_$_")[0] for file in file_list]))
    label_list = [int(csv_data[csv_data["vid_id"] == file]["view"].values[0] == 'off') for file in file_list]

    X_train, X_test, y_train, y_test = train_test_split(file_list, label_list, stratify=label_list,
                                                        test_size=test_split, random_state=3)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train,
                                                    test_size=val_split / (1 - test_split), random_state=3)
    
    file_list = os.listdir(directory)

    X_train = [file for file in tqdm(file_list) if file.split("_$_")[0] in X_train]
    X_val = [file for file in tqdm(file_list) if file.split("_$_")[0] in X_val]
    X_test = [file for file in tqdm(file_list) if file.split("_$_")[0] in X_test]

    y_train = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["view"].values[0] == 'off') 
            for file in tqdm(X_train)]
    y_val = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["view"].values[0] == 'off') 
            for file in tqdm(X_val)]
    y_test = [int(csv_data[csv_data["vid_id"] == file.split("_$_")[0]]["view"].values[0] == 'off') 
            for file in tqdm(X_test)]
    
    assert(len(X_train) + len(X_val) + len(X_test) == len(file_list))
    assert(len(y_train) + len(y_val) + len(y_test) == len(file_list))
    
    train_dataset = CustomDataset(directory, X_train, y_train, random_rotation)
    test_dataset = CustomDataset(directory, X_test, y_test, random_rotation)
    val_dataset = CustomDataset(directory, X_val, y_val, random_rotation)

    return train_dataset, test_dataset, val_dataset
