import torch
from Preprocessing import myDataset
from torch.utils.data import DataLoader

train_dataset_path = './train_dataset.csv'
val_dataset_path = './val_dataset.csv'
test_dataset_path = './test_dataset.csv'

train_dataset = myDataset(train_dataset_path)
val_dataset = myDataset(val_dataset_path)
test_dataset = myDataset(test_dataset_path)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)