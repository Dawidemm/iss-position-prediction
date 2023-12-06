import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mydataset import myDataset
from torchvision.transforms import ToTensor
from fetchdata import FetchData
from model import myModel, myLitModel

def draw_points_on_map(image, current_location, predicted_location):
    # Przeskaluj współrzędne do zakresu -180 do 180
    current_location_scaled = scale_coordinates(current_location)
    predicted_location_scaled = scale_coordinates(predicted_location)

    # Narysuj punkty na mapie
    cv2.circle(image, current_location_scaled, 5, (0, 255, 0), -1)  # Aktualna lokalizacja (kolor zielony)
    cv2.circle(image, predicted_location_scaled, 5, (0, 0, 255), -1)  # Przewidziana lokalizacja (kolor czerwony)

def scale_coordinates(coordinates):
    # Przeskaluj współrzędne do zakresu -180 do 180
    scaled_x = int(np.clip(coordinates[0], -180, 180))
    scaled_y = int(np.clip(coordinates[1], -180, 180))
    return scaled_x, scaled_y

def main():

    world_map = cv2.imread('project/world_map.jpg')

    torch_model = myModel()
    lit_model = myLitModel.load_from_checkpoint(
        checkpoint_path='lightning_logs/version_0/checkpoints/epoch=23-step=2640.ckpt',
        map_location=torch.device('cpu'),
        model=torch_model)
    
    i = 0
    while i < 10:

        fetch_data = FetchData()
        fetch_data = torch.tensor(list(map(float, next(fetch_data))), dtype=torch.float32)
        net_input = fetch_data/180

        net_output = lit_model(net_input)

        preds = net_output * 180

        print(fetch_data, preds)

        i += 1


    # # Przewiduj lokalizację dla każdego batcha
    # for batch in dataloader:
    #     input_data = batch['input']
    #     predicted_location = model(input_data)

    #     # Pobierz aktualne położenie z FetchData
    #     current_location = input_data.numpy()[0]

    #     # Narysuj punkty na mapie
    #     draw_points_on_map(world_map, current_location, predicted_location.detach().numpy()[0])

    # # Wyświetl mapę
    # cv2.imshow('World Map with Points', world_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
