import torch.nn as nn
import timm

# Define the FaceModel class
class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.eff_net = timm.create_model('efficientnet_b7', pretrained=False, num_classes=7)
        
    def forward(self, x):
        return self.eff_net(x)