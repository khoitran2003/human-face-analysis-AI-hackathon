import torchvision.models as models
import torch.nn as nn
import torch


class Modified_Resnet_50(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        del self.backbone.fc

        self.backbone.fc1 = nn.Linear(2048, 6)  # age:
        self.backbone.fc2 = nn.Linear(2048, 2)  # gender
        self.backbone.fc3 = nn.Linear(2048, 7)  # emotion

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        output1 = self.backbone.fc1(x)
        output2 = self.backbone.fc2(x)
        output3 = self.backbone.fc3(x)

        return output1, output2, output3


