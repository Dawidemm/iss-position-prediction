import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from GetData import FetchData
from Settings import SettingsMakeDataset
from Model import neural_network

model = neural_network()
model.build(input_shape=(1, 2))
model.load_weights('project/ModelWeights.h5')

data = FetchData(duration=SettingsMakeDataset.DURATION)
longitude, latitude = [], []
pred_longitude, pred_latitude = [], []

scaler_longitude = 6
scaler_latitude = 3

for i in range(SettingsMakeDataset.DURATION):
    lon, lat = next(data)
    lon, lat = np.float32(lon), np.float32(lat)
    longitude.append(lon), latitude.append(lat)

    lon_lat = np.array([lon, lat]).reshape(1, 2)
    preds = model.predict(lon_lat, verbose=0)
    pred_longitude.append(preds[:, 0]), pred_latitude.append(preds[:, 1])

def animate(i):
    ax1.cla()
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Longitude')
    ax1.plot(range(len(longitude[0:i])), longitude[0:i], linestyle='--')
    ax1.scatter(range(len(longitude[0:i])), longitude[0:i], label='Real Position')

    ax1.plot(range(len(longitude[0:i])), pred_longitude[0:i], linestyle='--')
    ax1.scatter(range(len(longitude[0:i])), pred_longitude[0:i], label='Predicted Position')
    #ax1.set_title(f'Longitude prediction r2: {r2_score(longitude, pred_longitude)}')
    ax1.legend()

    ax2.cla()
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Latitude')
    ax2.plot(range(len(latitude[0:i])), latitude[0:i], linestyle='--')
    ax2.scatter(range(len(latitude[0:i])), latitude[0:i], label='Real Position')

    ax2.plot(range(len(latitude[0:i])), pred_latitude[0:i], linestyle='--')
    ax2.scatter(range(len(latitude[0:i])), pred_latitude[0:i], label=f'Predicted Position')
    #ax2.set_title(f'Latitude prediction r2: {r2_score(latitude, pred_latitude)}')
    ax2.legend()

    # ax3.cla()
    # ax3.set_xlabel('Longitude')
    # ax3.set_ylabel('Latitude')
    # img = plt.imread('project/world_map.jpg')
    # ax3.imshow(img, extent=[-1080, 1080, -540, 540])
    # ax3.plot(longitude[0:i], latitude[0:i])
    # ax3.scatter(longitude[0:i], latitude[0:i])
    # ax3.plot(pred_longitude[0:i], pred_latitude[0:i], label='Real Position')
    # ax3.scatter(pred_longitude[0:i], pred_latitude[0:i], label='Predicted Position')

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 8))
ani = FuncAnimation(plt.gcf(), animate, interval=500, cache_frame_data=False)
plt.show()