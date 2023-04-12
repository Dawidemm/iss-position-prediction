# import tensorflow as tf
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from GetData import FetchData
from Settings import SettingsMakeDataset

data = FetchData(duration=SettingsMakeDataset.DURATION)
longitude, latitude = [], []
pred_longitude, pred_latitude = [], []

def animate(i):

    lon, lat = next(data)
    longitude.append(lon)
    latitude.append(lat)

    if len(longitude) > 9:
        longitude.pop(0), latitude.pop(0)

    ax1.cla()
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.plot(longitude, latitude, linestyle='--')
    ax1.scatter(longitude, latitude, label='Real Position')
    ax1.legend()


fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8))

ani = FuncAnimation(plt.gcf(), animate, interval=750, cache_frame_data=False)

plt.show()