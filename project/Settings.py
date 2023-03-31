from dataclasses import dataclass
import tensorflow as tf

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

@dataclass
class SettingsModel:
    '''
    Settings for the neural network model

    parameters:
    EPOCHS: int # Epochs of learning
    LEARNING_RATE: float # Learning rate
    EARLY_STOPPING: # tf.keras.callbacs object
    MODEL_CHECKPOINT: # tf.keras.callbacs object
    HISTORY: # tf.keras.callbacs object
    '''
    EPOCHS: int = 25
    LEARNING_RATE: float = 0.001
    EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(patience=5)
    MODEL_CHECKPOINT = tf.keras.callbacks.ModelCheckpoint('project/testModel.h5', save_best_only=True)
    HISTORY = tf.keras.callbacks.History()