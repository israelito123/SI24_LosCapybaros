import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet18, ResNet18_Weights

# El F sirve para aplicar simplemente operaciones como ReLU o el MaxPooling
# Por el otro lado, en los layers si tenemos pesos, por lo que se usa el nn.Module

file_path = pathlib.Path(__file__).parent.absolute()

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
    def __init__(self, input_dim: int, n_classes: int) -> None:  # Manera B
        super().__init__() 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #shape 1x48x48
    
        self.layer1 = nn.Conv2d(1,  out_channels=64,      kernel_size=3) # 46
        self.ReLU1 = nn.ReLU()
        

        self.layer2 = nn.Conv2d(64,         out_channels=128,     kernel_size=3) # 44
        self.ReLU2 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(kernel_size=2) #22

        self.layer3 = nn.Conv2d(128,        out_channels=128 ,    kernel_size=3) # 20
        self.ReLU3 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(kernel_size=2) # 10

        self.fc1 = nn.Linear(10 * 10 * 128, 1024)
        self.fc2 = nn.Linear(1024 , n_classes)
        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)

 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # TODO: Define la propagacion hacia adelante de tu red ✅
        
        x = self.layer1(x)
       # print(x.size())
        x = F.relu(x)
        x = self.layer2(x)
        #print(x.size())
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = self.layer3(x)
       # print(x.size())
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = torch.flatten(x , 1)
        #print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)



        #return x, logits, proba #Logits: Raw outputs from final layer, aqui habia return x
        return x
    
    def forward_inference(self, x: torch.Tensor) -> torch.Tensor: 

        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        logs = x

        return logs

    def predict(self, x):
        with torch.inference_mode():
            return self.forward_inference(x)

    def save_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        self.load_state_dict(torch.load(models_path))