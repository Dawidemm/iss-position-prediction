import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class myDataset(Dataset):
    def __init__(self, csv_file,):
        self.data = pd.read_csv(csv_file, header=0)
        # self.max_val = self.data.abs().max().max()
        self.max_val = 180
        self.num_samples = len(self.data)

    def __len__(self):
        return self.num_samples - 1

    def __getitem__(self, index):

        current_row = self.data.iloc[index]
        target = self.data.iloc[index + 1]

        current_row = torch.tensor([current_row['longitude'], current_row['latitude']], dtype=torch.float32) / self.max_val
        target = torch.tensor([target['longitude'], target['latitude']], dtype=torch.float32) / self.max_val
        
        return current_row, target