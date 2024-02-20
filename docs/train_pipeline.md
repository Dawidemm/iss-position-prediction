### - train_pipeline.py

This script defines a PyTorch Lightning training pipeline for a machine learning model using the `LightningLatLongPredictor` model, `LightningLatLongDatamodule` data module, and various PyTorch Lightning functionalities.

#### Usage

1. Run the script:

    ```bash
    python train_pipeline.py
    ```

#### Training Pipeline

The `train_pipeline` function performs the following steps:

1. Initializes a Lightning data module (`LightningLatLongDatamodule`) with training, validation, and testing datasets from specified CSV files (`TRAIN_DATASET_PATH`, `VAL_DATASET_PATH`, `TEST_DATASET_PATH`) and a specified batch size (`BATCH_SIZE`).

2. Initializes a Lightning model (`LightningLatLongPredictor`).

3. Configures a PyTorch Lightning Trainer with the following parameters:
   - `max_epochs`: Maximum number of training epochs (50 in this case).
   - `accelerator`: The accelerator is set to 'auto', allowing PyTorch Lightning to automatically choose the appropriate accelerator device based on the available hardware (CPU or GPU).
   - `callbacks`: Utilizes EarlyStopping callback to monitor the 'val_loss' and stop training if it does not improve within a patience of 5 epochs. Additionally, it uses the ModelCheckpoint callback to save the best model based on the 'val_loss' during training.
   - `logger`: The logger is set to False, indicating that no logging will be performed during training.

4. Fits the Lightning model to the training data using the provided data module.

5. Evaluates the trained model on the test data and prints the Mean Squared Error (MSE).

Note: Users can modify the paths to CSV files (`TRAIN_DATASET_PATH`, `VAL_DATASET_PATH`, `TEST_DATASET_PATH`) in the script to point to their generated datasets.