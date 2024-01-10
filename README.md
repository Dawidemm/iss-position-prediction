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
