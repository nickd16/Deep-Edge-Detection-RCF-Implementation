import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import os
import random

def display_images(images):
    fig, axs = plt.subplots(1, len(images))
    for i in range (len(images)):
        image, gray = images[i]
        if type(image) == torch.Tensor and image.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            image = ((image*std) + mean) * 255
        if type(image) == torch.Tensor: 
            image = image.permute(1, 2, 0)  
            image = image.int()
            image = image.numpy()
        if gray:
            axs[i].imshow(image, cmap='gray')
        else:
            axs[i].imshow(image)
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

class BSD(Dataset):
    def __init__(self, train):
        path = 'data/images/'
        path += 'train' if train else 'val'
        images = []
        edgemaps = []
        for file in os.listdir(path):
            img = cv.imread(path+'/'+file)
            images.append(img)
        path = 'data/groundTruth/'
        path += 'train' if train else 'val'
        for file in os.listdir(path):
            temp = path+'/'+file
            data = loadmat(temp)
            img_data = data['groundTruth']
            img = img_data[0][0]['Boundaries'][0][0]
            edgemaps.append(img) 
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.X = images
        self.Y = edgemaps
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x,y = self.X[idx], self.Y[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        if x.shape == (481, 321, 3):
            x = x.permute(2, 1, 0)
            y = y.permute(1, 0).unsqueeze(0)
        else:
            x = x.permute(2, 0, 1)
            y = y.unsqueeze(0)
        x = x / 255.0
        x = self.normalize(x)
        # if random.uniform(0,1) > 0.5:
        #     x = TF.hflip(x)
        #     y = TF.hflip(y)
        # if random.uniform(0,1) > 0.5:
        #     x = TF.vflip(x)
        #     y = TF.vflip(y)
        return (x,y)

def main():
    dataset = BSD(True)
    x, y = dataset[0]
    #display_images([(x,False),(y,True)]) 
    device = torch.device('cuda')  
    X = []
    Y = []
    for i in range (10):
        rand = int(random.uniform(1,399))
        x,y = dataset[rand]
        X.append(x)
        Y.append(y)
    X = torch.stack(X, dim=0)
    Y = torch.stack(Y, dim=0)
    print(X.shape)
    print(Y.shape)

if __name__ == '__main__':
    main()




