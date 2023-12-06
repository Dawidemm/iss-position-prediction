from dataclasses import dataclass
import tensorflow as tf

@dataclass
class SettingsMakeDataset:
    '''
    Settings for the MakeDataset class

    parameters:
    WHERE: str  # Where to save the dataset
    DURATION: int  # How many data points to fetch
    URL: str  # The url of the API
    '''
    DURATION: int = 25
    WHERE: str = 'dataset.csv'
    URL: str = 'http://api.open-notify.org/iss-now.json'

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
    MODEL_CHECKPOINT = tf.keras.callbacks.ModelCheckpoint('project/ModelWeights.h5', save_best_only=True)
    HISTORY = tf.keras.callbacks.History()