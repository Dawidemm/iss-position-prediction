import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class PreprocessingTrainingDataset():
    def __init__(self, dataset: pd.DataFrame):

        '''
        Preprocess the dataset for the training, validation and test purposes

        parameters:
        dataset: pd.DataFrame

        methods:
        set_target: set the target for the dataset where the target is the next position of the ISS
        standard_scaled: standard scaled the dataset
        train: split the dataset into train dataset and train target
        val: split the dataset into validation dataset and validation target
        test: split the dataset into test dataset and test target
        '''

        self.dataset = dataset
        self.dataset.columns = ['longitude', 'latitude']
        self.lenght = len(self.dataset)

    def set_target(self):

        self.dataset['future_longitude'] = self.dataset['longitude'].shift(periods=1)
        self.dataset['future_latitude'] = self.dataset['latitude'].shift(periods=1)
        self.dataset = self.dataset.drop(self.dataset.index[0])
        self.dataset = self.dataset.drop(self.dataset.index[-1])

        return self
    
    def standard_scaled(self):

        self.standard_scaler = StandardScaler()
        self.dataset = np.array(self.dataset)

        self.target = self.dataset[:, [2, 3]]
        self.dataset = self.dataset[:, [0, 1]]
        
        self.dataset = self.standard_scaler.fit_transform(self.dataset)
        self.target = self.standard_scaler.transform(self.target)

        return self
    
    def train(self) -> tuple:

        self.train_dataset = self.dataset[0:int(self.lenght * 0.7)]
        self.train_target = self.target[0:int(self.lenght * 0.7)]

        return self.train_dataset, self.train_target
    
    def val(self) -> tuple:

        self.val_dataset = self.dataset[int(self.lenght * 0.7):int(self.lenght * 0.9)]
        self.val_target = self.target[int(self.lenght * 0.7):int(self.lenght * 0.9)]

        return self.val_dataset, self.val_target
    
    def test(self) -> tuple:

        self.test_dataset = self.dataset[int(self.lenght * 0.9):]
        self.test_target = self.target[int(self.lenght * 0.9):]

        return self.test_dataset, self.test_target