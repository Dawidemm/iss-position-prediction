import json
import requests
from dataclasses import dataclass

@dataclass
class ApiKeyConfig:
    '''
    A dataclass for managing API key configurations.

    Attributes:
    - URL (str): The default URL for fetching International Space Station (ISS) position data.
    '''
    
    URL: str = 'http://api.open-notify.org/iss-now.json'

class FetchData():
    '''
    A class for fetching and providing real-time geographical coordinates of the International Space Station (ISS).

    Args:
    - url (str, optional): The URL for fetching ISS position data. Defaults to ApiKeyConfig.URL.

    Methods:
    - __next__(): Fetches the next geographical coordinates of the ISS.
    - __iter__(): Returns the iterator object.
    '''

    def __init__(self, url: str = ApiKeyConfig.URL):
        self.url = url

    def __next__(self) -> tuple:

        answer = requests.get(self.url)
        answer = json.loads(answer.text)
        position = answer['iss_position']

        return position['longitude'], position['latitude']

    def __iter__(self):
        return self