import torch
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def create_gaussian_blob(x, y, sigma, block_size):
    x_vals = torch.arange(2 * block_size)
    y_vals = torch.arange(2 * block_size)
    xx, yy = torch.meshgrid(x_vals, y_vals)
    gaussian_blob = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    gaussian_blob = gaussian_blob / torch.max(gaussian_blob)
    return gaussian_blob


def create_grid(x1, y1, x2, y2, x3, y3, x4, y4, sigma=2, block_size=20):
    t1 = create_gaussian_blob(x1, y1, sigma=sigma, block_size=block_size)
    t2 = create_gaussian_blob(x2 + block_size, y2, sigma=sigma, block_size=block_size)
    t3 = create_gaussian_blob(x3, y3 + block_size, sigma=sigma, block_size=block_size)
    t4 = create_gaussian_blob(x4 + block_size, y4 + block_size, sigma=sigma, block_size=block_size)
    return t1 + t2 + t3 + t4


class CustomDataset(Dataset):

    def __init__(self, len, block_size, sigma):
        super(CustomDataset, self).__init__()
        self.len = len
        self.block_size = block_size
        self.sigma = sigma


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        result = create_grid(
            *self.get_class_rand_tensor(idx), 
            sigma=self.sigma, 
            block_size=self.block_size)
        return result, idx
    

    def get_class_rand_tensor(self, idx):

        if idx % 2 == 0:
            return torch.cat((torch.randint(0, self.block_size, (4,)), 
                            torch.randint(0, self.block_size // 2, (1,)), 
                            torch.randint(0, self.block_size, (1,)), 
                            torch.randint(0, self.block_size // 2, (1,)), 
                            torch.randint(0, self.block_size, (1,))))
        
        return torch.cat((torch.randint(0, self.block_size, (4,)), 
                            torch.randint(self.block_size // 2, self.block_size, (1,)), 
                            torch.randint(0, self.block_size, (1,)), 
                            torch.randint(self.block_size // 2, self.block_size, (1,)), 
                            torch.randint(0, self.block_size, (1,))))
    

def get_dataloaders(len, batch_size=32, train_workers=4, block_size=20, sigma=2):
    train_dataset = CustomDataset(len, block_size=block_size, sigma=sigma)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_workers)
    return train_dataloader