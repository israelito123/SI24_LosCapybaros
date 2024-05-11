import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from network import Network
import torch
from utils import to_numpy, get_transforms, add_img_text
from dataset import EMOTIONS_MAP
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path)
    val_transforms, unnormalize = get_transforms("test", img_size = 48)
    tensor_img = val_transforms(img)
    denormalized = unnormalize(tensor_img)
    return img, tensor_img, denormalized

def predict(img_title_paths):
    '''
        Hace la inferencia de las imagenes
        args:
        - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    '''
    # Cargar el modelo
    modelo = Network(48, 7)
    modelo.load_model("modelo_1.pt")
    for path in img_title_paths:
        # Cargar la imagen
        # np.ndarray, torch.Tensor
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)

        # Inferencia
        proba = modelo.predict(transformed)
        pred = torch.argmax(proba, -1).item()
        pred_label = EMOTIONS_MAP[pred]

        # Original / transformada
        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label}")

        # Mostrar la imagen
        denormalized = to_numpy(denormalized)
        denormalized = cv2.resize(denormalized, (resize_value, resize_value))
        cv2.imshow("Predicción - original", img)
        cv2.imshow("Predicción - transformed", denormalized)
        cv2.waitKey(0)

if __name__=="__main__":
    # Direcciones relativas a este archivo


    img_paths = ["./test_imgs/30.jpg","./test_imgs/31.jpg",
            "./test_imgs/29.jpg","./test_imgs/28.jpg",
            "./test_imgs/27.jpg","./test_imgs/26.jpg",
            "./test_imgs/25.jpg","./test_imgs/24.jpg",
            "./test_imgs/23.jpg","./test_imgs/22.jpg",
            "./test_imgs/20.jpg","./test_imgs/19.jpg",
            "./test_imgs/17.jpg","./test_imgs/21.jpg",
            "./test_imgs/16.jpg","./test_imgs/15.jpg",
            "./test_imgs/14.jpg","./test_imgs/13.jpg",
            "./test_imgs/12.jpg","./test_imgs/11.jpg",
            "./test_imgs/10.jpg","./test_imgs/9.jpg",
            "./test_imgs/8.jpg","./test_imgs/7.jpg",
            "./test_imgs/6.jpg","./test_imgs/5.jpg",
            "./test_imgs/4.jpg","./test_imgs/2.jpg",
            "./test_imgs/1.jpg","./test_imgs/happy.png",
            "./test_imgs/happy2.png","./test_imgs/happy3.png",
            "./test_imgs/mad.png","./test_imgs/neutral.png",
            "./test_imgs/neutral2.png","./test_imgs/scared.png"]
    

    predict(img_paths)
