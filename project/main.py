import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mydataset import myDataset
from fetchdata import FetchData
from model import myModel, myLitModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

EARTH_RADIUS = 6371
ISS_ORBIT_RADIUS = 420
animation_paused = False

def polar_to_cartesian(lon, lat, radius):
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z

def draw_earth(ax):
    phi, theta = np.mgrid[0.0:2.0*np.pi:100j, 0.0:np.pi:50j]
    x = EARTH_RADIUS * np.sin(theta)*np.cos(phi)
    y = EARTH_RADIUS * np.sin(theta)*np.sin(phi)
    z = EARTH_RADIUS * np.cos(theta)
    ax.plot_surface(x, y, z, color='green', alpha=0.2, linewidth=0)

def draw_iss(ax, longitude, latitude):
    x, y, z = polar_to_cartesian(longitude, latitude, EARTH_RADIUS + ISS_ORBIT_RADIUS)
    ax.scatter(x, y, z, c='blue', marker='o', label='ISS')

def draw_predictions(ax, preds):
    global prediction_legend_added

    for i in range(len(preds)):
        longitude, latitude = preds[i].detach().numpy()
        x, y, z = polar_to_cartesian(longitude, latitude, EARTH_RADIUS + ISS_ORBIT_RADIUS)
        if i == 0:
            ax.scatter(x, y, z, c='orange', marker='o', label='Predictions')
        else:
            ax.scatter(x, y, z, c='orange', marker='o')

    if not prediction_legend_added:
        ax.legend()
        prediction_legend_added = True

def update_plot(frame, model, ax, longitude, latitude, preds):
    global animation_paused
    ax.cla()

    fetch_data = FetchData()
    fetch_data = torch.tensor(list(map(float, next(fetch_data))), dtype=torch.float32)
    net_input = fetch_data / 180

    net_output = model(net_input)
    pred = net_output * 180

    lon = fetch_data[0].item()
    lat = fetch_data[1].item()

    if len(longitude) == 0 or (lon != longitude[-1] or lat != latitude[-1]):
        longitude.append(lon)
        latitude.append(lat)
        preds.append(pred)

    # x, y, z = polar_to_cartesian(longitude, latitude, EARTH_RADIUS+ISS_ORBIT_RADIUS)

    draw_earth(ax)
    draw_iss(ax, longitude, latitude)
    draw_predictions(ax, preds)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if animation_paused:
        ani.event_source.stop()
    else:
        ani.event_source.start()

    ax.legend()

def pause_animation(event):
    global animation_paused
    animation_paused = not animation_paused

def main():
    global ani, prediction_legend_added

    torch_model = myModel()
    lit_model = myLitModel.load_from_checkpoint(
        checkpoint_path='lightning_logs/version_0/checkpoints/epoch=23-step=2640.ckpt',
        map_location=torch.device('cpu'),
        model=torch_model)

    longitude = []
    latitude = []
    preds = []
    prediction_legend_added = False

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ani = FuncAnimation(fig, update_plot, fargs=(lit_model, ax, longitude, latitude, preds), frames=100, interval=500)

    fig.canvas.mpl_connect('key_press_event', pause_animation)

    plt.show()

if __name__ == "__main__":
    main()