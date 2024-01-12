import os
import numpy as np
import matplotlib.pyplot as plt

EARTH_RADIUS = 6371
ISS_ORBIT_RADIUS = 420

def polar_to_cartesian(lon, lat, radius):
    '''
    Convert polar coordinates (longitude, latitude) to Cartesian coordinates (x, y, z).

    Parameters:
    - lon (float): Longitude in degrees.
    - lat (float): Latitude in degrees.
    - radius (float): The radius from the origin to the point in space.

    Returns:
    Tuple[float, float, float]: Cartesian coordinates (x, y, z).

    The function takes longitude, latitude, and radius as input and converts
    them to Cartesian coordinates. The resulting coordinates represent a point
    in 3D space on the surface of a sphere centered at the origin.

    Example:
    x, y, z = polar_to_cartesian(45.0, 30.0, 10.0)
    '''

    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    return x, y, z

def draw_earth(ax, earth_radius=EARTH_RADIUS):
    '''
    Draw a 3D representation of the Earth on the specified Axes.

    Parameters:
    - ax (matplotlib.axes._subplots.Axes3D): The 3D Axes on which to draw on.

    Description:
    This function generates a 3D plot of the Earth using a spherical coordinate system.
    It uses the provided Axes object to draw the Earth's surface as a solid sphere.
    The surface color is set to green with some transparency (alpha=0.2), and the linewidth is set to 0.

    Note:
    This function assumes the global variable EARTH_RADIUS is defined and represents the radius of the Earth.

    Example:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_earth(ax)
    plt.show()
    '''

    phi, theta = np.mgrid[0.0:2.0*np.pi:100j, 0.0:np.pi:50j]
    x = earth_radius * np.sin(theta)*np.cos(phi)
    y = earth_radius * np.sin(theta)*np.sin(phi)
    z = earth_radius * np.cos(theta)
    ax.plot_surface(x, y, z, color='green', alpha=0.2, linewidth=0)

def draw_points(ax, data, label, earth_radius=EARTH_RADIUS, iss_orbit_radius=ISS_ORBIT_RADIUS):
    '''
    Scatter plot points on a 3D Axes.

    Parameters:
    - ax (matplotlib.axes._subplots.Axes3D): The 3D Axes to draw on.
    - data (list): List of 2D tensors containing longitude and latitude coordinates.
    - label (str): Label for the points, should be either 'Pred' or 'True'.
    - earth_radius (float, optional): Earth's radius. Default is the global variable EARTH_RADIUS.
    - iss_orbit_radius (float, optional): Radius of the ISS orbit. Default is the global variable ISS_ORBIT_RADIUS.

    The function takes 3D coordinates (longitude, latitude) from the input data
    and converts them to Cartesian coordinates. It then scatter plots these
    points on the provided 3D Axes with different colors based on the label.
    The last point is marked with a cross.

    Example:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = torch.tensor([[45.0, 30.0], [60.0, 40.0], [70.0, 50.0]])
    draw_points(ax, data, label='True')
    plt.show()
    '''

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
        x, y, z = polar_to_cartesian(longitude, latitude, earth_radius + iss_orbit_radius)

        if i == 0:
            ax.scatter(x, y, z, c=color, marker='o', label=label)
        elif i < (len(data)-1):
            ax.scatter(x, y, z, c=color, marker='o')
        else:
            ax.scatter(x, y, z, c=color_last, marker='x')

def check_model_version(checkpoints_dir='src/checkpoints'):
    '''
    Check the number of model versions in the specified directory.

    Parameters:
    - checkpoints_dir (str): Directory containing model checkpoints.

    Returns:
    - str: A string representation of the number of model versions in the directory.
           If the directory doesn't exist, '0' is returned.
           Otherwise, the count of model versions is returned as a string.
    '''
    
    if not os.path.exists(checkpoints_dir):
        return f'0'
    
    else:
        models = os.listdir(checkpoints_dir)

        return f'{len(models)}'
    
def get_last_model_version(checkpoints_dir='src/checkpoints'):
    '''
    Get the path to the last version of the model from the specified directory.

    Parameters:
    - checkpoints_dir (str): Directory containing model checkpoints.

    Returns:
    - model_path (str): Path to the last model checkpoint.
    '''

    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Directory not found: {checkpoints_dir}")

    files = os.listdir(checkpoints_dir)

    checkpoint_files = [file for file in files if file.endswith('.ckpt')]

    checkpoint_files.sort()

    if not checkpoint_files:
        raise FileNotFoundError(f"No model checkpoints found in directory: {checkpoints_dir}")

    last_checkpoint = checkpoint_files[-1]

    last_checkpoint_path = os.path.join(checkpoints_dir, last_checkpoint)

    return last_checkpoint_path