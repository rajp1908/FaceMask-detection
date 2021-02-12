import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from dataset import Dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

class Evaluate:
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_batch_size = 100
        self.classes = ['without_mask', 'with_mask', 'non_humans']
        self.y_true = []
        self.y_pred = []
        self.test_loader = torch.utils.data.DataLoader(test_dataset, self.test_batch_size,
                                                shuffle=False)

    def run(self,mode=None):      
         
        print('-------------------- test evaluation')
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(3)]
            n_class_samples = [0 for i in range(3)]
            for images, labels in self.test_loader:
                outputs = self.model(images)
                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                self.y_true += labels.tolist()
                self.y_pred += predicted.tolist()
                
                for i in range(min(self.test_batch_size,len(labels))):
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            print('')
            print(f'Accuracy of the network: {acc} %')
            print('')
            
            accuracy = []
            for i in range(3):
                accuracy.append(100.0 * n_class_correct[i] / n_class_samples[i])

        precision, recall, fscore, support = precision_recall_fscore_support(self.y_true, self.y_pred)
        print('           ', tuple(self.classes))
        print('accuracy:   ' +',     '.join(['%10.2f']*len(accuracy)) % tuple(accuracy))
        print('precision:  ' +',     '.join(['%10.2f']*len(accuracy)) % tuple(precision))
        print('recall:     ' +',     '.join(['%10.2f']*len(accuracy)) % tuple(recall))
        print('fscore:     ' +',     '.join(['%10.2f']*len(accuracy)) % tuple(fscore))
        print('support:    ' +',     '.join(['%10.2f']*len(accuracy)) % tuple(support))

        if (mode == "cm"):
            print('')
            print('confusion matrix: ')
            cm = confusion_matrix(self.y_true, self.y_pred)
            
            cm_sum = np.sum(cm, axis=1, keepdims=True)
            cm_perc = cm / cm_sum.astype(float) * 100
            
            print(cm)
            print('')
            print('confusion matrix in percentage: ')
            print(cm_perc)
            
            df_cm = pd.DataFrame(cm_perc, index = [i for i in self.classes],
                    columns = [i for i in self.classes])
            sn.heatmap(df_cm, annot=True )
            plt.show()
