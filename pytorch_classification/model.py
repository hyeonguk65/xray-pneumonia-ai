import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2, pretrained=True):

    # 1. ResNet18 불러오기 (사전학습 가중치)
    model = models.resnet18(pretrained=pretrained)

    # 2. 마지막 FC 레이어를 2클래스 출력으로 교체
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model
