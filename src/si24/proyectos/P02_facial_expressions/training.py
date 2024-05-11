
from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
import network
from tqdm import tqdm
from dataset import get_loader
from network import Network
from plot_losses import PlotLosses


def validation_step(val_loader, net, cost_function):
    '''
        Realiza un epoch completo en el conjunto de validación
        args:
        - val_loader (torch.DataLoader): dataloader para los datos de validación
        - net: instancia de red neuronal de clase Network
        - cost_function (torch.nn): Función de costo a utilizar

        returns:
        - val_loss (float): el costo total (promedio por minibatch) de todos los datos de validación
    '''
    val_loss = 0.0
    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch['transformed']
        batch_imgs = batch_imgs.cuda()
        batch_labels = batch['label']
        batch_labels = batch_labels.cuda()
        device = net.device
        batch_labels = batch_labels.to(device)
        with torch.inference_mode():
            # TODO: realiza un forward pass, calcula el loss y acumula el costo
            cost = net.forward(batch_imgs)
            val_loss += cost_function(cost, batch_labels)
            val_loss = val_loss.cuda()

    # TODO: Regresa el costo promedio por minibatch
    return val_loss/len(val_loader)

def train():
    # Hyperparametros
    learning_rate = 5e-3
    n_epochs=25
    batch_size = 512

    # Train, validation, test loaders
    train_dataset, train_loader = \
        get_loader("train",
                    batch_size=batch_size,
                    shuffle=True)
    val_dataset, val_loader = \
        get_loader("val",
                    batch_size=batch_size,
                    shuffle=False)
    print(f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}")

    plotter = PlotLosses()
    # Instanciamos tu red
    modelo = Network(input_dim=1,
                     n_classes = 7)

    # TODO: Define la funcion de costo
    criterion = nn.CrossEntropyLoss()

    # Define el optimizador
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate,betas=(0.9,0.999),eps=4e-4)

    best_epoch_loss = np.inf
    mejorCosto = 10e20
    for epoch in range(n_epochs):
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch['transformed']
            batch_imgs = batch_imgs.cuda()
            batch_labels = batch['label']
            batch_labels = batch_labels.cuda()
            # TODO Zero grad, forward pass, backward pass, optimizer step
            optimizer.zero_grad()
            yhat = modelo(batch_imgs)

            cost = criterion(yhat, batch_labels)
            cost.backward()

            optimizer.step()

            # TODO acumula el costo
            train_loss += cost.item()

        # TODO Calcula el costo promedio
        train_loss = train_loss/len(train_loader)
        train_loss = train_loss
        val_loss = validation_step(val_loader, modelo, criterion)
        tqdm.write(f"Epoch: {epoch}, train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

        # TODO guarda el modelo si el costo de validación es menor al mejor costo de validación
       
        if (val_loss < mejorCosto):
            mejorCosto = val_loss
            modelo.save_model(model_name="Entrenado")
            print("Se ha guardado un nuevo modelo")

        val_loss = torch.Tensor.numpy(val_loss, force = True) 
        plotter.on_epoch_end(train_loss, val_loss)
    plotter.on_train_end()

if __name__=="__main__":
    train()