## LatLongDataset (dataset_module.py)

### Description:
The `LatLongDataset` class, defined in the `dataset_module.py` file, serves as a wrapper for geolocation data, facilitating its preparation for use in a model. Data is loaded from a CSV file and processed to remove duplicates.

### Methods:

#### `__init__(self, csv_file: str, step: int)`
- `csv_file`: Path to the CSV file containing geolocation data.
- `step`: Number of time steps between training data and target data.

#### `__len__(self) -> int`
Returns the number of available examples in the dataset.

#### `__getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]`
Returns a tuple containing training data and its corresponding target data. The data is normalized and transformed into PyTorch tensors.

## LightningLatLongDatamodule (dataset_module.py)

### Description:
The `LightningLatLongDatamodule` class, defined in the `dataset_module.py` file, inherits from `LightningDataModule` and is used to easily prepare data for training, validation, and testing of a model.

### Methods:

#### `__init__(self, train_csv: str, val_csv: str, test_csv: str, batch_size: int)`
- `train_csv`: Path to the CSV file with training data.
- `val_csv`: Path to the CSV file with validation data.
- `test_csv`: Path to the CSV file with test data.
- `batch_size`: Batch size used in the DataLoaders.

#### `setup(self, stage: str) -> None`
Initializes training, validation, and test datasets. In this method, the `LatLongDataset` class is utilized for creating instances of training, validation, and test datasets.

#### `train_dataloader(self) -> DataLoader`
Returns a DataLoader for training data.

#### `val_dataloader(self) -> DataLoader`
Returns a DataLoader for validation data.

#### `test_dataloader(self) -> DataLoader`
Returns a DataLoader for test data.

## LatLongPredictor (predictor_module.py)

### Description:
The `LatLongPredictor` class, defined in the `predictor_module.py` file, is a simple neural network for geolocation prediction. It consists of three linear layers.

### Methods:

#### `__init__(self)`
Initializes the neural network with three linear layers.

#### `forward(self, x: torch.Tensor) -> torch.Tensor`
Performs a forward pass through the network.

## LightningLatLongPredictor (predictor_module.py)

### Description:
The `LightningLatLongPredictor` class, defined in the `predictor_module.py` file, is a LightningModule for training and evaluation of the `LatLongPredictor` model.

### Methods:

#### `__init__(self, model=LatLongPredictor(), metric=torchmetrics.MeanSquaredError())`
- `model`: An instance of the `LatLongPredictor` model.
- `metric`: The metric used for evaluation during training.

#### `forward(self, x: torch.Tensor) -> torch.Tensor`
Performs a forward pass through the model.

#### `_shared_step(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
Shared step for training, validation, and testing. Computes loss and metrics.

#### `training_step(self, batch, batch_idx) -> torch.Tensor`
Training step. Computes and logs training loss and metric.

#### `validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]`
Validation step. Computes and logs validation loss and metric.

#### `test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]`
Test step. Computes and logs test loss and metric.

#### `configure_optimizers(self) -> torch.optim.Optimizer`
Configures the optimizer. Returns an instance of the Adam optimizer.