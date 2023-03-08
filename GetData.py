import json
import requests
import time

URL = 'http://api.open-notify.org/iss-now.json'

class FetchData():
    def __init__(self, duration: int, url: str) -> None:
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
    def __init__(self, duration: int) -> None:
        self.duration = duration
    
    def save_as_csv(self) -> None:
        trainig_dataset = open('training_dataset_test.csv', 'w')
        data_generator = FetchData(duration=self.duration, url=URL)
        
        for _ in range(self.duration):
            position = next(data_generator)
            trainig_dataset.write(f'{position[0]}, {position[1]}\n')


        return None
    
t1 = time.time()
training_data = MakeDataset(duration=50000)
training_data.save_as_csv()
t2 = time.time()

t = t2 - t1