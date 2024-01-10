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