import json
import requests
from dataclasses import dataclass

@dataclass
class SettingsMakeDataset:
    '''
    Settings for the MakeDataset class

    parameters:
    WHERE: str  # Where to save the dataset
    URL: str  # The url of the API
    '''
    WHERE: str = 'dataset.csv'
    URL: str = 'http://api.open-notify.org/iss-now.json'

class FetchData():
    def __init__(self, url: str = SettingsMakeDataset.URL) -> None:
        '''
        A class for fetching and providing real-time geographical coordinates of the International Space Station (ISS).

        Args:
        - url (str, optional): The URL for fetching ISS position data. Defaults to SettingsMakeDataset.URL.

        Methods:
        - __next__(): Fetches the next geographical coordinates of the ISS.
        - __iter__(): Returns the iterator object.
        '''
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
    def __init__(self, type: str, duration: int) -> None:
        self.duration = duration
        self.type = type

    def save_as_csv(self) -> None:
        trainig_dataset = open(f'{self.type}_{SettingsMakeDataset.WHERE}', 'w')
        trainig_dataset.write(f'longitude,latitude\n')

        data_generator = FetchData()
        
        for _ in range(self.duration):
            position = next(data_generator)
            trainig_dataset.write(f'{position[0]},{position[1]}\n')

        return None


def make_dataset(type: str, duration: int):
    dataset = MakeDataset(type=type, duration=duration)
    dataset.save_as_csv()

if __name__ == '__main__':
    make_dataset(type='val', duration=10000)