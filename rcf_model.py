import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torchinfo
import time

# Organization:
# Conv1_1 -> Conv5_3 are the normal vgg16 convolutional layers
# Conv21_1_1 -> Conv21_5_3 are the convolutional layers that decrease filter size to 21
# The 21 conv outputs are added element-wise and mapped to a single filter image for each block
# The 5 single filter image mappings are fused with a 1x1 conv for a fused image
# * The 4 lower layers are upscaled with bilinear interpolation for the fused image
# Final images are passed through sigmoid activation to obtain edge prediction maps

# The model organization follows the above order for the 5 vgg blocks 

class RCF(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.sigmoid = nn.Sigmoid()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        #----------------------------------------------------------------------------------#

        self.conv21_1_1 = nn.Conv2d(in_channels=64, out_channels=21, kernel_size=1)
        self.conv21_1_2 = nn.Conv2d(in_channels=64, out_channels=21, kernel_size=1)
        self.conv1_1_1  = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1)

        self.conv21_2_1 = nn.Conv2d(in_channels=128, out_channels=21, kernel_size=1)
        self.conv21_2_2 = nn.Conv2d(in_channels=128, out_channels=21, kernel_size=1)
        self.conv1_1_2  = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1)  

        self.conv21_3_1 = nn.Conv2d(in_channels=256, out_channels=21, kernel_size=1)
        self.conv21_3_2 = nn.Conv2d(in_channels=256, out_channels=21, kernel_size=1)
        self.conv21_3_3 = nn.Conv2d(in_channels=256, out_channels=21, kernel_size=1)
        self.conv1_1_3  = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1)

        self.conv21_4_1 = nn.Conv2d(in_channels=512, out_channels=21, kernel_size=1)
        self.conv21_4_2 = nn.Conv2d(in_channels=512, out_channels=21, kernel_size=1)
        self.conv21_4_3 = nn.Conv2d(in_channels=512, out_channels=21, kernel_size=1)
        self.conv1_1_4  = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1)

        self.conv21_5_1 = nn.Conv2d(in_channels=512, out_channels=21, kernel_size=1)
        self.conv21_5_2 = nn.Conv2d(in_channels=512, out_channels=21, kernel_size=1)
        self.conv21_5_3 = nn.Conv2d(in_channels=512, out_channels=21, kernel_size=1)
        self.conv1_1_5  = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=1)

        self.convf      = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)


    def forward(self, x):
        conv1_1_out = self.relu(self.conv1_1(x))
        conv1_2_out = self.relu(self.conv1_2(conv1_1_out))
        x = self.maxpool(conv1_2_out)

        conv2_1_out = self.relu(self.conv2_1(x))
        conv2_2_out = self.relu(self.conv2_2(conv2_1_out))
        x = self.maxpool(conv2_2_out)

        conv3_1_out = self.relu(self.conv3_1(x))
        conv3_2_out = self.relu(self.conv3_2(conv3_1_out))
        conv3_3_out = self.relu(self.conv3_3(conv3_2_out))
        x = self.maxpool(conv3_3_out)

        conv4_1_out = self.relu(self.conv4_1(x))
        conv4_2_out = self.relu(self.conv4_2(conv4_1_out))
        conv4_3_out = self.relu(self.conv4_3(conv4_2_out))
        x = self.maxpool(conv4_3_out)

        conv5_1_out = self.relu(self.conv5_1(x))
        conv5_2_out = self.relu(self.conv5_2(conv5_1_out))
        conv5_3_out = self.relu(self.conv5_3(conv5_2_out))

        #---------------------------------------------------------------#

        conv21_1_1 = self.conv21_1_1(conv1_1_out)
        conv21_1_2 = self.conv21_1_2(conv1_2_out)
        eltwise1 = conv21_1_1+conv21_1_2
        conv1_1_1 = self.conv1_1_1(eltwise1)

        conv21_2_1 = self.conv21_2_1(conv2_1_out)
        conv21_2_2 = self.conv21_2_2(conv2_2_out)
        eltwise2 = conv21_2_1+conv21_2_2
        conv1_1_2 = self.conv1_1_2(eltwise2)

        conv21_3_1 = self.conv21_3_1(conv3_1_out)
        conv21_3_2 = self.conv21_3_2(conv3_2_out)
        conv21_3_3 = self.conv21_3_3(conv3_3_out)
        eltwise3 = conv21_3_1+conv21_3_2+conv21_3_3
        conv1_1_3 = self.conv1_1_3(eltwise3)

        conv21_4_1 = self.conv21_4_1(conv4_1_out)
        conv21_4_2 = self.conv21_4_2(conv4_2_out)
        conv21_4_3 = self.conv21_4_3(conv4_3_out)
        eltwise4 = conv21_4_1+conv21_4_2+conv21_4_3
        conv1_1_4 = self.conv1_1_4(eltwise4)

        conv21_5_1 = self.conv21_5_1(conv5_1_out)
        conv21_5_2 = self.conv21_5_2(conv5_2_out)
        conv21_5_3 = self.conv21_5_3(conv5_3_out)
        eltwise5 = conv21_5_1+conv21_5_2+conv21_5_3
        conv1_1_5 = self.conv1_1_5(eltwise5)

        outputs = [conv1_1_1, conv1_1_2, conv1_1_3, conv1_1_4, conv1_1_5]
        for i in range(1, len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=(321,481), mode='bilinear')
        fused = torch.stack(outputs, dim=1).squeeze(2)
        fused = self.convf(fused)
        for i in range(len(outputs)):
            outputs[i] = self.sigmoid(outputs[i])
        return outputs, self.sigmoid(fused)
    
def main():
    device = torch.device('cuda')
    model = RCF().to(device)
    model.load_state_dict(torch.load('weights/initial_weights.pth'))
    tensor = torch.randn(10,3,321,481).to(device)
    outputs, fused = model(tensor)
    for i in outputs:
        print(i.shape)
    print(fused.shape)

if __name__ == '__main__':
    main()
