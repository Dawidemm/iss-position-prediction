import numpy as np
import matplotlib.pyplot as plt

EARTH_RADIUS = 6371
ISS_ORBIT_RADIUS = 420

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

def draw_points(ax, data, label):

    if label == 'Pred':
        color = 'red'
        color_last = 'orange'
    elif label == 'True':
        color = 'blue'
        color_last = 'cyan'
    else:
        raise ValueError(f'Pick one from ["Preds", "True"].')

    for i in range(len(data)):

        longitude, latitude = data[i].detach().numpy()
        x, y, z = polar_to_cartesian(longitude, latitude, EARTH_RADIUS + ISS_ORBIT_RADIUS)

        if i == 0:
            ax.scatter(x, y, z, c=color, marker='o', label=label)
        elif i < (len(data)-1):
            ax.scatter(x, y, z, c=color, marker='o')
        else:
            ax.scatter(x, y, z, c=color_last, marker='o', label=f'Last {label}')