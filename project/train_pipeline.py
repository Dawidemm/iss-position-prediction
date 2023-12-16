import torch
import lightning as pl
from mydataset import myDataset
from torch.utils.data import DataLoader
from model import DataStep, myModel, myLitModel
from lightning.pytorch.callbacks import EarlyStopping

torch.manual_seed(10)
    
train_dataset_path = './train_dataset.csv'
val_dataset_path = './val_dataset.csv'
test_dataset_path = './test_dataset.csv'

DATA_STEP = DataStep.step

train_dataset = myDataset(train_dataset_path, step=DATA_STEP)
val_dataset = myDataset(val_dataset_path, step=DATA_STEP)
test_dataset = myDataset(test_dataset_path, step=DATA_STEP)

def train_pipeline() -> None:

    '''
    Train a PyTorch Lightning model using a custom training pipeline.

    This function performs the following steps:
    1. Creates data loaders for training, validation, and testing datasets.
    2. Initializes a PyTorch model (`myModel`) and wraps it with a Lightning model (`myLitModel`).
    3. Configures a PyTorch Lightning Trainer with specified parameters:
       - `max_epochs`: Maximum number of training epochs (50 in this case).
       - `accelerator`: Auto-select accelerator device for training.
       - `callbacks`: Utilizes EarlyStopping callback to stop training if validation loss does not improve.
       - `enable_checkpointing`: Enables model checkpointing during training.
    4. Fits the Lightning model to the training data using the provided data loaders.
    5. Evaluates the trained model on the test data and prints the Mean Squared Error (MSE).
    6. Saves the PyTorch model (`torch_model`) to a file ('litmodel.pt').

    Returns:
        None
    '''

    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    torch_model = myModel()
    lit_model = myLitModel(torch_model)

    trainer = pl.Trainer(
        max_epochs=50, 
        accelerator='auto', 
        callbacks=[EarlyStopping('val_loss', patience=5)], 
        enable_checkpointing=True)

    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    test_mse = trainer.test(dataloaders=test_dataloader, ckpt_path='best')[0]

    model_path = 'litmodel.pt'
    torch.save(torch_model, model_path)

    return None

if __name__ == '__main__':
    train_pipeline()