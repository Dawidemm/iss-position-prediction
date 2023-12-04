import torch
from torch import nn
import lightning as pl

class myModel(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [nn.Linear(2, 10),
                  nn.Linear(10, 10),
                  nn.Linear(10, 2)]
        
        self.net = nn.Sequential(*layers)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = myModel()
print(list(model.parameters()))