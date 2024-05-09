import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet18, ResNet18_Weights

file_path = pathlib.Path(__file__).parent.absolute()

def calc_out_dims(input_dim, kernel_size, stride=1, padding=0):
    out_dim = (input_dim -  kernel_size + 2*padding)//stride
    out_dim += 1
    return out_dim

def build_backbone(model='resnet18', weights='imagenet', freeze=True, last_n_layers=2):
    if model == 'resnet18':
        backbone = resnet18(pretrained=weights == 'imagenet')
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone
    else:
        raise Exception(f'Model {model} not supported')

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TODO: Define las capas de tu red
        output_dims = 32
        for kernel_size in [5,5,5]:
           output_dims = calc_out_dims(output_dims, kernel_size)
        print(output_dims)

        self.net1 = nn.Sequential(
                nn.Conv2d(3, out_channels=16, kernel_size=3), # B, 16, 30, 30
                nn.ReLU(),
                nn.Conv2d(16, out_channels=32, kernel_size=3), # B, 32, 28, 28
                nn.ReLU(),
                nn.Conv2d(32, out_channels=64, kernel_size=6), # B, 64, 23, 23, // B, linear_feat
                nn.ReLU(),
                nn.Flatten(), # B, 64 * 23 *23
                nn.Linear(64 * 23 * 23, 64), # B, 64
                nn.ReLU(),
                nn.Linear(64, 10) # B, 10
                )
        self.to(self.device)

        # TODO: Calcular dimension de salida
    def calc_out_dims(input_dim, kernel_size, stride=1, padding=0):
        out_dim = (input_dim -  kernel_size + 2*padding)//stride
        out_dim += 1
        return out_dim

   
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim

    def __init__(self):
        super().__init__()
        # TODO: define las capas de tu red
        self.conv1 = nn.Conv2d(3, 16, 3) # B, 16, 30, 30
        self.fc1 = nn.Linear(16 * 30 * 30, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        x = self.net1(x) 
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)

        logits = self.net(x)
        proba = nn.Softmax(logits)
        
        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        '''
            Guarda el modelo en el path especificado
            args:
            - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
            - path (str): path relativo donde se guardará el modelo
        '''
        models_path = file_path / 'models' / model_name
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.net.state_dic)

    def load_model(self, model_name: str):
        '''
            Carga el modelo en el path especificado
            args:
            - path (str): path relativo donde se guardó el modelo
        '''
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / 'models' / model_name
        self.net.load_state_dict(torch.load(models_path))
