import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

class myDataset(Dataset):
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
    
class myLitDataModule(LightningDataModule):
    def __init__(self, train_csv: str, val_csv: str, test_csv, batch_size: int):
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        self.train_dataset = myDataset(csv_file=self.train_csv, step=1)
        self.val_dataset = myDataset(csv_file=self.val_csv, step=1)
        self.test_dataset = myDataset(csv_file=self.test_csv, step=1)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)  
    

if __name__ == '__main__':

    train_dataset_path = './train_dataset.csv'
    train_dataset = myDataset(train_dataset_path, step=1)
    print(len(train_dataset.data))

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    X, y = next(iter(train_dataloader))

    print(X.shape)
    print(y.shape)