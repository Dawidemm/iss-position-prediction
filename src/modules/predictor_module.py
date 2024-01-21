import torch
from torch import nn
import lightning as pl


class LatLongPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [nn.Linear(2, 16),
                  nn.Linear(16, 16),
                  nn.Linear(16, 2)]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        return self.net(x)
    
class LightningLatLongPredictor(pl.LightningModule):
    def __init__(self, model=LatLongPredictor()):
        super().__init__()
        self.model = model
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
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, outputs = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss, y, outputs = self._shared_step(batch, batch_idx)

        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return optimizer