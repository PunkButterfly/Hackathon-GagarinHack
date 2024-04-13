import torch
import torch.nn as nn
from PIL import Image

from torchvision.models import resnet18
from torchvision import transforms

import torch.nn.functional as F
import os

def soft_max(array):
    return nn.Softmax(dim=1)(array)


class ModelV1:
    def __init__(self):
        model = resnet18(pretrained=True)
        # Замораживаем параметры модели, чтобы не обучать их заново
        for param in model.parameters():
            param.requires_grad = False

        # Заменяем последний слой модели на новый слой с количеством классов равным количеству классов в нашем датасете
        model.fc = nn.Linear(model.fc.in_features, 4)

        self.model = model

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Размер изображения
            transforms.ToTensor()  # Преобразование в тензор
        ])
        
        self.idx_to_class ={0: 'Drivers', 1: 'PTS', 2: 'Passports', 3: 'STS'}

class ModelV2:
    def __init__(self):

        out_dim = 4

        encoder = resnet18(pretrained=True)
        model = EmbedNet(encoder, out_dim)

        self.model = model

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Размер изображения
            transforms.ToTensor()  # Преобразование в тензор
        ])
        
        self.idx_to_class ={0: 'Drivers', 1: 'PTS', 2: 'Passports', 3: 'STS'}

class EmbedNet(nn.Module):
        def __init__(self, base_model, out_dim):
            super(EmbedNet, self).__init__()
            self.base_model = base_model
            self.base_model.fc = torch.nn.Linear(base_model.fc.in_features, 512)
            self.fc1 = torch.nn.Linear(512, 512)
            self.fc2 = torch.nn.Linear(512, 256)
            self.fc3 = torch.nn.Linear(256, out_dim)

        def forward(self, x):
            x = self.base_model(x)
            x = self.fc1(F.normalize(x))
            x = self.fc2(F.normalize(x))
            x = self.fc3(F.normalize(x))
            return F.normalize(x)

class ModelV3:
    def __init__(self):

        out_dim = 7

        encoder = resnet18(pretrained=True)
        model = EmbedNet(encoder, out_dim)

        self.model = model

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Размер изображения
            transforms.ToTensor()  # Преобразование в тензор
        ])
        
        self.idx_to_class = {
            0: 'Drivers',
            1: 'Drivers-2',
            2: 'PTS',
            3: 'Passports',
            4: 'Passports-2',
            5: 'STS',
            6: 'STS-2'
            }
        
class Classifier:
    def __init__(self, path_to_weights, weights_name, device = 'cpu', ModelClass=ModelV3):
        self.device = device

        self.model_class = ModelClass()
        self.model = self.model_class.model

        self.model.load_state_dict(torch.load(os.path.join(path_to_weights, weights_name), map_location=torch.device('cpu')))

    def process_img(self, img):
        self.model.eval()

        transformed_img = self.model_class.transform(img)


        predict = self.model(transformed_img.to(self.device).unsqueeze(0))
        parsed_predict = predict.cpu().detach().numpy().tolist()[0]
        print(predict)
        self.model_class.idx_to_class

        result = {
            self.model_class.idx_to_class[i]: parsed_predict[i]
            for i in range(len(parsed_predict))
        }
        return result