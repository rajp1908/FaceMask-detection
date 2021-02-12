import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from dataset import Dataset
import time

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # CNN.custom_cnn_init(self)
        CNN.alexnet_cnn_init(self,'m')
        # CNN.alexnet_cnn_init(self,'o')

    def forward(self, x):   
        # x = CNN.custom_cnn_forward(self,x)
        x = CNN.alexnet_cnn_forward(self,x,'m')
        # x = CNN.alexnet_cnn_forward(self,x,'o')
        return x

class CNN:
    def __init__(self):
        self.model = None
        self.average_accuracy = None

    @staticmethod
    def custom_cnn_init(self):
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 3)
    
    @staticmethod
    def custom_cnn_forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(F.relu(self.conv2(x))) 
        x = F.relu(self.conv3(x)) 
        x = self.pool(F.relu(self.conv4(x))) 
        x = x.view(-1, 64 * 16 * 16) 
        nn.Dropout(p=0.5)            
        x = F.relu(self.fc1(x)) 
        nn.Dropout(p=0.5) 
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x   
    
    @staticmethod
    def alexnet_cnn_init(self,resolution):
        if (resolution == 'm'):
            self.conv1 = nn.Conv2d(3, 27, 3, stride=2, padding=2)
            self.conv2 = nn.Conv2d(27, 71, 3, padding=2)
            self.conv3 = nn.Conv2d(71, 107, 3, padding=1)
            self.conv4 = nn.Conv2d(107, 107, 3, padding=1)
            self.conv5 = nn.Conv2d(107, 71, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(71 * 9 * 9, 1000)
            self.fc2 = nn.Linear(1000, 500)
            self.fc3 = nn.Linear(500, 3)
        elif (resolution == 'o'):
            self.conv1 = nn.Conv2d(3, 96, 3, stride=2, padding=2)
            self.conv2 = nn.Conv2d(96, 256, 3, padding=2)
            self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
            self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
            self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(256 * 9 * 9, 1000)
            self.fc2 = nn.Linear(1000, 500)
            self.fc3 = nn.Linear(500, 3)
    
    @staticmethod
    def alexnet_cnn_forward(self, x, resolution): 
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # print(x.shape)
        if (resolution == 'm'):
            x = x.view(-1, 71 * 9 * 9) 
        elif (resolution == 'o'):
            x = x.view(-1, 256 * 9 * 9) 
        nn.Dropout(p=0.5)         
        x = F.relu(self.fc1(x))  
        nn.Dropout(p=0.5)         
        x = F.relu(self.fc2(x))             
        x = self.fc3(x) 
        return x                  

    def train(self, train_dataset):

        num_epochs = 4
        train_batch_size = 32
        learning_rate = 0.001

        # create loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                                shuffle=True)
        self.model = ConvNet()  
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        start_time = time.time()

        loss_list = []
        acc_list = []
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total = labels.size(0)
                _,predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()

                loss_list.append(loss.item())
                acc_list.append(correct/total)

                if (i+1) % math.floor(total_step / 6) == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {(correct/total)*100:.2f}%')

        # print("training executaion time:  %s seconds" % (time.time() - start_time))
        print('')
        print('-------------------- training evaluation')
        print('')
        self.average_loss = sum(loss_list)/len(loss_list)
        print('average loss: ', self.average_loss)
        self.average_accuracy = sum(acc_list)/len(acc_list)
        print('average accuracy: ', self.average_accuracy)
        print('')

        # save the model
        PATH = './cnn.pt'
        torch.save(self.model.state_dict(), PATH)
                     



