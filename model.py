import torch
import torch.nn as nn
from torchvision import models
class model_resnet(nn.Module):
    def __init__(self, n_input, n_class,  n_hidden, dropout ):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size= (7, 7) ,stride = (2, 2), padding = (3, 3), bias = False)
        self.resnet.fc = nn.Identity()
        output_size = 512

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(output_size, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(n_hidden, n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(self.resnet(x))