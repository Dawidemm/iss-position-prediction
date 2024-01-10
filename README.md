## its_data_fetcher.py (modules folder)

### Description:
The `its_data_fetcher.py` module, located in the `modules` folder, contains a class and a dataclass for fetching real-time geographical coordinates of the International Space Station (ISS) through an API.

### Classes:

#### `ApiKeyConfig`
A dataclass for managing API key configurations.

##### Attributes:
- `URL` (str): The default URL for fetching International Space Station (ISS) position data.

#### `FetchData`
A class for fetching and providing real-time geographical coordinates of the ISS.

##### Args:
- `url` (str, optional): The URL for fetching ISS position data. Defaults to `ApiKeyConfig.URL`.

##### Methods:
- `__next__():` Fetches the next geographical coordinates of the ISS.
- `__iter__():` Returns the iterator object.

### Usage:
To use this module, create an instance of the `FetchData` class and call `__next__()` to retrieve the next ISS coordinates.

Example:
```python
from its_data_fetcher import FetchData

# Create an instance of FetchData
fetcher = FetchData()

# Fetch the next ISS coordinates
coordinates = next(fetcher)
print("Current ISS Coordinates:", coordinates)

## dataset_module.py (modules folder)

### Description:
The `dataset_module.py` module, located in the `modules` folder, provides a dataset class (`LatLongDataset`) and a PyTorch Lightning DataModule class (`LightningLatLongDatamodule`) for handling geolocation data.

### Classes:

#### `LatLongDataset`
A PyTorch Dataset class for handling geolocation data.

##### Args:
- `csv_file` (str): Path to the CSV file containing geolocation data.
<!-- - `step` (int): Number of time steps between training data and target data. -->

##### Methods:
- `__init__(self, csv_file: str, step: int):` Initializes the dataset with the specified CSV file and time step.
- `__len__(self) -> int:` Returns the number of available examples in the dataset.
- `__getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:` Returns a tuple containing training data and its corresponding target data.

#### `LightningLatLongDatamodule`
A PyTorch Lightning DataModule class for handling geolocation datasets.

##### Args:
- `train_csv` (str): Path to the CSV file with training data.
- `val_csv` (str): Path to the CSV file with validation data.
- `test_csv` (str): Path to the CSV file with test data.
- `batch_size` (int): Batch size used in the DataLoaders.

##### Methods:
- `__init__(self, train_csv: str, val_csv: str, test_csv: str, batch_size: int):` Initializes the DataModule with the paths to training, validation, and test datasets.
- `setup(self, stage: str) -> None:` Initializes training, validation, and test datasets.
- `train_dataloader(self) -> DataLoader:` Returns a DataLoader for training data.
- `val_dataloader(self) -> DataLoader:` Returns a DataLoader for validation data.
- `test_dataloader(self) -> DataLoader:` Returns a DataLoader for test data.

### Usage:
To use this module, create instances of `LatLongDataset` for your data and `LightningLatLongDatamodule` for handling datasets during training, validation, and testing.
```python
from dataset_module import LightningLatLongDatamodule

# Specify paths to CSV files
train_csv = 'path/to/train_dataset.csv'
val_csv = 'path/to/val_dataset.csv'
test_csv = 'path/to/test_dataset.csv'

# Set batch size
batch_size = 512

# Create a LightningLatLongDatamodule instance
datamodule = LightningLatLongDatamodule(train_csv, val_csv, test_csv, batch_size)

# Access DataLoader for training data
train_dataloader = datamodule.train_dataloader()

## predictor_module.py (modules folder)

### Description:
The `predictor_module.py` module, located in the `modules` folder, defines a neural network (`LatLongPredictor`) for geolocation prediction and a PyTorch Lightning module (`LightningLatLongPredictor`) for training, validation, and testing of the model.

### Classes:

#### `DataStep`
A dataclass for managing the step parameter in the model.

##### Attributes:
- `step` (int): The step parameter. Defaults to 1.

#### `LatLongPredictor`
A PyTorch Module class representing a neural network for geolocation prediction.

##### Methods:
- `__init__(self):` Initializes the network with three linear layers.
- `forward(self, x: torch.Tensor) -> torch.Tensor:` Performs a forward pass through the network.

#### `LightningLatLongPredictor`
A PyTorch Lightning Module class for training, validation, and testing the `LatLongPredictor` model.

##### Args:
- `model` (LatLongPredictor, optional): An instance of the `LatLongPredictor` model.
- `metric` (torchmetrics.Metric, optional): The metric used for evaluation during training.

##### Methods:
- `__init__(self, model=LatLongPredictor(), metric=torchmetrics.MeanSquaredError()):` Initializes the Lightning module with the specified model and metric.
- `forward(self, x: torch.Tensor) -> torch.Tensor:` Performs a forward pass through the model.
- `_shared_step(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:` Shared step for training, validation, and testing. Computes loss and metrics.
- `training_step(self, batch, batch_idx) -> torch.Tensor:` Training step. Computes and logs training loss and metric.
- `validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:` Validation step. Computes and logs validation loss and metric.
- `test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:` Test step. Computes and logs test loss and metric.
- `configure_optimizers(self) -> torch.optim.Optimizer:` Configures the optimizer. Returns an instance of the Adam optimizer.

### Usage:
To use this module, create an instance of `LightningLatLongPredictor` for model training and evaluation.

Example:
```python
from predictor_module import LightningLatLongPredictor

# Create an instance of LightningLatLongPredictor
predictor = LightningLatLongPredictor()

# Train the model using the provided training pipeline
predictor.train_pipeline()

# Evaluate the trained model on the test dataset
predictor.test()