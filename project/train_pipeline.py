import torch
import lightning as pl
from mydataset import myLitDataModule
from model import myModel, myLitModel
from lightning.pytorch.callbacks import EarlyStopping

torch.manual_seed(10)
    
TRAIN_DATASET_PATH = 'datasets/test_dataset.csv'
VAL_DATASET_PATH = 'datasets/val_dataset.csv'
TEST_DATASET_PATH = 'datasets/test_dataset.csv'
BATCH_SIZE = 512

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
    4. Fits the Lightning model to the training data using the provided datamodule.
    5. Evaluates the trained model on the test data and prints the Mean Squared Error (MSE).

    Returns:
        None
    '''

    datamodule = myLitDataModule(train_csv=TRAIN_DATASET_PATH,
                                 val_csv=VAL_DATASET_PATH,
                                 test_csv=TEST_DATASET_PATH,
                                 batch_size=BATCH_SIZE)

    lit_model = myLitModel()

    trainer = pl.Trainer(
        max_epochs=50, 
        accelerator='auto', 
        callbacks=[EarlyStopping('val_loss', patience=5)], 
        enable_checkpointing=True)

    trainer.fit(lit_model, datamodule=datamodule)

    test_mse = trainer.test(datamodule=datamodule, ckpt_path='best')[0]

    return None

if __name__ == '__main__':
    train_pipeline()