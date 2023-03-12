from dataclasses import dataclass

@dataclass
class SettingsMakeDataset:
    '''
    Settings for the MakeDataset class

    parameters:
    where: str  # Where to save the dataset
    duration: int  # How many data points to fetch
    url: str  # The url of the API
    '''
    duration: int = 25
    where: str = 'dataset.csv'
    url: str = 'http://api.open-notify.org/iss-now.json'