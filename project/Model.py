import tensorflow as tf
import pandas as pd
from Preprocessing import PreprocessingTrainingDataset
from Settings import SettingsModel

def build_and_compile_model() -> tf.keras.models.Sequential:

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=2))
    model.add(tf.keras.layers.Dense(units=64))
    model.add(tf.keras.layers.Dense(units=64))
    model.add(tf.keras.layers.Dense(units=2))

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(SettingsModel.LEARNING_RATE),
        metrics=['accuracy']
    )

    return model

def model_fit(dataset: pd.DataFrame) -> tf.keras.models.Sequential:

    train_dataset, train_target = dataset.train()
    val_dataset, val_target = dataset.val()

    model = build_and_compile_model()

    model.fit(train_dataset, train_target, 
            epochs=SettingsModel.EPOCHS, 
            validation_data=(val_dataset, val_target), 
            callbacks=[SettingsModel.EARLY_STOPPING, SettingsModel.MODEL_CHECKPOINT, SettingsModel.HISTORY],
            verbose=0
    )
    return model

if __name__ == '__main__':

    dataset = pd.read_csv('project/training_dataset.csv')
    dataset = PreprocessingTrainingDataset(dataset=dataset)
    dataset.set_target()
    dataset.standard_scaled()

    model = model_fit(dataset=dataset)