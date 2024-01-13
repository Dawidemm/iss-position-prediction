import torch
from torch import nn
import lightning as pl
import torchmetrics
from dataclasses import dataclass

@dataclass
class DataStep:
    step: int = 1

class LatLongPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [nn.Linear(2, 16 * DataStep.step),
                  nn.Linear(16 * DataStep.step, 16 * DataStep.step),
                  nn.Linear(16 * DataStep.step, 2)]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        return self.net(x)
    
class LightningLatLongPredictor(pl.LightningModule):
    def __init__(self, model=LatLongPredictor(), metric=torchmetrics.MeanSquaredError()):
        super().__init__()
        self.model = model
        self.metric = metric
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def _shared_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)

        return loss, y, outputs
    
    def training_step(self, batch, batch_idx):
        loss, y, outputs = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss)

        self.train_metric = self.metric
        self.train_metric(outputs, y)
        self.log('train_mse', self.train_metric.compute(), prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, outputs = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss)

        self.val_metric = self.metric
        self.val_metric(outputs, y)
        self.log('val_mse', self.val_metric.compute(), prog_bar=True)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss, y, outputs = self._shared_step(batch, batch_idx)

        self.test_metric = self.metric
        self.test_metric(outputs, y)
        self.log('test_mse', self.val_metric.compute())

        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return optimizer