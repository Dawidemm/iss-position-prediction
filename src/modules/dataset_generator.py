from modules.iss_data_fetcher import FetchData

class GenerateDataset():
    '''
    A class for generating and saving training datasets with geographical coordinates.

    Args:
    - type (str): The typeof the dataset.
    - duration (int): The duration, number of data points, to be generated and saved.

    Methods:
    - save_as_csv(): Fetches geographical coordinates using a data generator and saves them as a csv file'.
    '''

    def __init__(self, type: str, duration: int):
        self.duration = duration
        self.type = type

    def save_as_csv(self):

        trainig_dataset = open(f'datasets/{self.type}_dataset.csv', 'w')
        trainig_dataset.write(f'longitude,latitude\n')

        data_generator = FetchData()
        
        for _ in range(self.duration):
            position = next(data_generator)
            trainig_dataset.write(f'{position[0]},{position[1]}\n')