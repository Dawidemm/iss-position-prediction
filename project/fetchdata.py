import json
import requests
from dataclasses import dataclass

@dataclass
class SettingsMakeDataset:
    '''
    A dataclass for storing attributes related to dataset creation.
    
    Attributes:
    - WHERE (str): The default path for saving datasets.
    - URL (str): The default URL for fetching ISS position data.
    '''
    WHERE: str = ''
    URL: str = 'http://api.open-notify.org/iss-now.json'

    def set_path(self, path):
        self.WHERE = f'datasets/{path}_dataset.csv'

class FetchData():
    '''
    A class for fetching and providing real-time geographical coordinates of the International Space Station (ISS).

    Args:
    - url (str, optional): The URL for fetching ISS position data. Defaults to SettingsMakeDataset.URL.

    Methods:
    - __next__(): Fetches the next geographical coordinates of the ISS.
    - __iter__(): Returns the iterator object.
    '''

    def __init__(self, url: str = SettingsMakeDataset.URL):
        self.url = url

    def __next__(self) -> tuple:

        answer = requests.get(self.url)
        answer = json.loads(answer.text)
        position = answer['iss_position']

        return position['longitude'], position['latitude']

    def __iter__(self):
        return self
    
class MakeDataset():
    '''
    A class for generating and saving training datasets with geographical coordinates.

    Args:
    - type (str): The typeof the dataset.
    - duration (int): The duration, number of data points, to be generated and saved.

    Methods:
    - save_as_csv(): Fetches geographical coordinates using a data generator and saves them
      as a CSV file named '{type}_{SettingsMakeDataset.WHERE}'.
    '''

    def __init__(self, type: str, duration: int):
        self.duration = duration
        self.type = type

    def save_as_csv(self):
        settings = SettingsMakeDataset()
        settings.set_path(self.type)

        trainig_dataset = open(f'{settings.WHERE}', 'w')
        trainig_dataset.write(f'longitude,latitude\n')

        data_generator = FetchData()
        
        for _ in range(self.duration):
            position = next(data_generator)
            trainig_dataset.write(f'{position[0]},{position[1]}\n')

def make_dataset(type: str, duration: int):
    dataset = MakeDataset(type=type, duration=duration)
    dataset.save_as_csv()


if __name__ == '__main__':
    make_dataset(type='abc', duration=10)