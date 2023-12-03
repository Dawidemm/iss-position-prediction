import json
import requests
from Settings import SettingsMakeDataset

class FetchData():
    def __init__(self, duration: int = SettingsMakeDataset.DURATION, url: str = SettingsMakeDataset.URL) -> None:
        '''
        Fetch data from the API and return the longitude and latitude of the ISS

        parameters:
        duration: int
        url: str
        '''
        self.duration = duration
        self.url = url

    def __next__(self) -> tuple:

        for _ in range(self.duration):

            answer = requests.get(self.url)
            answer = json.loads(answer.text)
            position = answer['iss_position']

            return position['longitude'], position['latitude']

    def __iter__(self):
        return self
class MakeDataset():
    '''
    Make a dataset from the data fetched from the API and save it as a csv file

    parameters:
    duration: int
    '''
    def __init__(self, duration: int = SettingsMakeDataset.DURATION) -> None:
        self.duration = duration

    def save_as_csv(self) -> None:
        trainig_dataset = open(SettingsMakeDataset.WHERE, 'w')
        trainig_dataset.write(f'longitude,latitude\n')

        data_generator = FetchData(self.duration)
        
        for _ in range(self.duration):
            position = next(data_generator)
            trainig_dataset.write(f'{position[0]},{position[1]}\n')

        return None


if __name__ == '__main__':
    dataset = MakeDataset(duration=25000)
    dataset.save_as_csv()