import torch
from torch import nn
import lightning as pl
from Preprocessing import myDataset
from torch.utils.data import DataLoader

class myModel(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [nn.Linear(2, 10),
                  nn.Linear(10, 10),
                  nn.Linear(10, 2)]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class myLitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer
    
train_dataset_path = './train_dataset.csv'
val_dataset_path = './val_dataset.csv'
test_dataset_path = './test_dataset.csv'

train_dataset = myDataset(train_dataset_path)
val_dataset = myDataset(val_dataset_path)
test_dataset = myDataset(test_dataset_path)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

torch_model = myModel()
lit_model = myLitModel(torch_model)

trainer = pl.Trainer(max_epochs=10)
trainer.fit(lit_model, train_dataloaders=train_dataloader)