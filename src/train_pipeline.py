import torch
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from modules.dataset_module import LightningLatLongDatamodule
from modules.predictor_module import LightningLatLongPredictor
from modules.utils import get_model_version

torch.manual_seed(10)
    
TRAIN_DATASET_PATH = 'datasets/test_dataset.csv'
VAL_DATASET_PATH = 'datasets/val_dataset.csv'
TEST_DATASET_PATH = 'datasets/test_dataset.csv'
BATCH_SIZE = 128
SEQUENCE_LENGHT = 1
MAX_EPOCHS = 50

def train_pipeline():

    '''
    Train a PyTorch Lightning model using a custom training pipeline.

    This function performs the following steps:
    1. Creates a datamodule with training, validation, and testing datasets.
    2. Initializes a Lightning model (`LightningLatLongPredictor`).
    3. Configures a PyTorch Lightning Trainer with the following parameters:
       - `max_epochs`: Maximum number of training epochs (50 in this case).
       - `accelerator`: The accelerator is set to 'auto', allowing PyTorch Lightning to automatically choose the appropriate accelerator device based on the available hardware (CPU or GPU).
       - `callbacks`: Utilizes EarlyStopping callback to monitor the 'val_loss' and stop training if it does not improve within a patience of 5 epochs. Additionally, it uses the ModelCheckpoint callback to save the best model based on the 'val_loss' during training.
       - `logger`: The logger is set to False, indicating that no logging will be performed during training.

    4. Fits the Lightning model to the training data using the provided datamodule.
    5. Evaluates the trained model on the test data and prints the Mean Squared Error (MSE).
    '''

    datamodule = LightningLatLongDatamodule(
        train_csv=TRAIN_DATASET_PATH,
        val_csv=VAL_DATASET_PATH, 
        test_csv=TEST_DATASET_PATH, 
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGHT
    )
    
    datamodule.setup(stage='train')

    lit_model = LightningLatLongPredictor()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=0.001,
        patience=5
    )
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        dirpath='src/checkpoints/',
        filename=f'model=v{get_model_version()}' + '-{epoch:02d}-{val_loss:.4e}' + f'-batch_size={BATCH_SIZE}'
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS, 
        accelerator='auto', 
        callbacks=[early_stopping, checkpoint_callback],
        logger=False
    )

    trainer.fit(lit_model, datamodule=datamodule)

    datamodule.setup(stage='test')

    test_mse = trainer.test(datamodule=datamodule, ckpt_path='best')[0]


if __name__ == '__main__':
    train_pipeline()