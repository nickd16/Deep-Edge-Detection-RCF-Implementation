from BSDS500 import BSD
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from rcf_model import RCF
from rcf_loss import RCFLoss, accuracy
from BSDS500 import display_images

def main():
    device = torch.device('cuda')

    model = RCF().to(device)
    model.load_state_dict(torch.load('weights/current_weights.pth'))

    train_dataset = BSD(True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataset = BSD(False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    epochs = 100
    lr = 5e-4
    criterion = RCFLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.1)
    min_loss = 10000000
    avg_loss = 0
    avg_acc = 0

    for i in range(epochs):
        for bidx, (x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            outputs, fused = model(x)

            # optimizer.zero_grad()
            # loss = torch.zeros(1).to(device)
            # for output in outputs:
            #     loss = loss + criterion(output, y)
            # loss = loss + criterion(fused, y)
            # loss = loss / (len(outputs)+1)
            # acc = accuracy(fused, y)
            # loss.backward()
            # optimizer.step()
            # scheduler.step()

            if i % 20 == 0:
                tensor = fused.cpu().detach()[0]
                tensor = (tensor >= .6)
                tensor2 = y.cpu().detach()[0]
                display_images([(x.squeeze(0).cpu().detach(),False),(tensor, True),(tensor2, True)])

        # for bidx, (x,y) in enumerate(val_loader):
        #     x = x.to(device)
        #     y = y.to(device)
        #     outputs, fused = model(x)
        #     loss = torch.zeros(1).to(device)
        #     for output in outputs:
        #         loss = loss + criterion(output, y)
        #     loss = loss + criterion(fused, y)
        #     loss = loss / (len(outputs)+1)    
        #     avg_loss += (loss.item())
        #     avg_acc += (acc.item())

        # avg_loss = avg_loss/20
        # avg_acc = avg_acc/20
        # print(f'Epoch {i+1} | Loss {round(avg_loss,4)} | Accuracy {round(avg_acc,4)}')
        # if avg_loss < min_loss:
        #     torch.save(model.state_dict(), 'weights/current_weights.pth')
        #     min_loss = avg_loss
        # avg_loss = 0
        # avg_acc = 0

if __name__ == '__main__':
    main()
