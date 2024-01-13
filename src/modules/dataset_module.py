import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

torch.manual_seed(10)

class LatLongDataset(Dataset):
    def __init__(self, csv_file: str, step: int):
        self.data = pd.read_csv(csv_file, header=0)
        self.data = self.data.drop_duplicates()
        self.step = step
        self.max_val = 180
        self.num_samples = len(self.data)

    def __len__(self):
        return self.num_samples - self.step - 1

    def __getitem__(self, index):

        current_row = self.data.iloc[index:index+self.step].values
        target = self.data.iloc[index+self.step+1].values

        current_row  = torch.tensor(current_row, dtype=torch.float32) /self.max_val
        target = torch.tensor(target, dtype=torch.float32) /self.max_val
        target = torch.reshape(target, (1, 2))
        
        return current_row, target
    
class LightningLatLongDatamodule(LightningDataModule):
    def __init__(self, train_csv: str, val_csv: str, test_csv, batch_size: int):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = LatLongDataset(csv_file=self.train_csv, step=1)
            self.val_dataset = LatLongDataset(csv_file=self.val_csv, step=1)
        if stage == 'test':
            self.test_dataset = LatLongDataset(csv_file=self.test_csv, step=1)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)