from dataset import Dataset
from cnn import CNN
from cnn import ConvNet
from evaluate import Evaluate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from dataset import Dataset
import time

# cross validation function
def crossValidation(dataset, k_fold):
    
    dataset_size = len(dataset)
    fold = int(dataset_size / k_fold)
    validation_accuracy = []
    for i in range(k_fold):
        print('')
        print('*** cross validation itteration: ', i+1)
        print('')
        
        train_indexes = list(range(0,i * fold)) + list(range(i * fold + fold,dataset_size))
        validation_indexes = list(range(i * fold,i * fold + fold))
        
        train_dataset = torch.utils.data.dataset.Subset(dataset,train_indexes)
        validation_dataset = torch.utils.data.dataset.Subset(dataset,validation_indexes) 

        # create cnn and train the model with dataset
        conv_net = CNN()
        conv_net.train(train_dataset)
        validation_accuracy.append(conv_net.average_accuracy)
        #  evalute multiple performnace factores of the cnn
        eval = Evaluate(conv_net.model, validation_dataset)
        eval.run() 
    
    print('')
    print('>>> cross validation average accuracy: ', sum(validation_accuracy)/len(validation_accuracy))
    print('')






#  create dataset
dataset = Dataset() 
train_dataset,test_dataset = dataset.get_dataset()

# K-fold cross validation
crossValidation(dataset=train_dataset, k_fold=5)

# create cnn and train the model with dataset
print('')
print('*** main training process starts')
print('')
conv_net = CNN()
conv_net.train(train_dataset)

# PATH = './cnn.pt'
# trained_model = ConvNet()
# trained_model.load_state_dict(torch.load(PATH))

#  evalute multiple performnace factores of the cnn
eval = Evaluate(conv_net.model, test_dataset)
eval.run("cm") 