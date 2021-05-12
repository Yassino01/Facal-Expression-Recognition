from __future__ import print_function
import argparse
import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt


from dataloader import Plain_Dataset, eval_data_dataloader
from model import Deep_Emotion


DATA_PATH = "./data"

def Train(epochs,train_loader,val_loader,criterion,optmizer,device):
    
  #logs values for train and loss 

  train_epoch_loss = []
  train_epoch_acc = []

  val_epoch_loss = []
  val_epoch_acc = []
  '''
  Training Loop
  '''
  print("===================================Start Training===================================")
  for e in range(epochs):
    train_loss = 0
    validation_loss = 0
    train_correct = 0
    val_correct = 0
    # Train the model  #
    net.train()
    
    for data, labels in train_loader:
      data, labels = data.to(device), labels.to(device)
      optmizer.zero_grad()
      outputs = net(data)
      loss = criterion(outputs,labels)
      loss.backward()
      optmizer.step()
      train_loss += loss.item()
      #print("Batch loss item ", loss.item())
      _, preds = torch.max(outputs,1)
      train_correct += torch.sum(preds == labels.data)

    #validate the model#
    
    net.eval()
    for data,labels in val_loader:
      data, labels = data.to(device), labels.to(device)
      val_outputs = net(data)
      val_loss = criterion(val_outputs, labels)
      validation_loss += val_loss.item()
      _, val_preds = torch.max(val_outputs,1)
      val_correct += torch.sum(val_preds == labels.data)

    train_loss = train_loss/len(train_dataset)
    train_acc = train_correct.double() / len(train_dataset)
    validation_loss =  validation_loss / len(validation_dataset)
    val_acc = val_correct.double() / len(validation_dataset)
    print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}%'
                                                        .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))

    train_epoch_loss.append(train_loss)
    train_epoch_acc.append(train_acc)
    val_epoch_loss.append(validation_loss)
    val_epoch_acc.append(val_acc)

  torch.save(net.state_dict(),'deep_emotion-{}-{}-{}.pt'.format(epochs,batchsize,lr))
  return train_epoch_acc, train_epoch_loss, val_epoch_acc, val_epoch_loss
  print("===================================Training Finished===================================")



if __name__ == '__main__':

  net = Deep_Emotion()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  net.to(device)
  
  print("Model Summary : ", net)
  print("device : ", device)
  
  traincsv_file = DATA_PATH+'/'+'finaltrain.csv'
  validationcsv_file = DATA_PATH+'/'+'val.csv'
  train_img_dir = DATA_PATH+'/'+'finaltrain/'
  validation_img_dir = DATA_PATH+'/'+'val/'

  batchsize = 128
  epochs = 200

  transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
  train_dataset= Plain_Dataset(csv_file=traincsv_file, img_dir = train_img_dir, datatype = 'finaltrain', transform = transformation)
  validation_dataset= Plain_Dataset(csv_file=validationcsv_file, img_dir = validation_img_dir, datatype = 'val', transform = transformation)
  train_loader= DataLoader(train_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
  val_loader=   DataLoader(validation_dataset,batch_size=batchsize,shuffle = True,num_workers=0)

  criterion= nn.CrossEntropyLoss()
  optmizer= optim.Adam(net.parameters())
  train_acc, train_loss, val_acc, val_loss = Train(epochs, train_loader, val_loader, criterion, optmizer, device)

  plt.figure()
  plt.plot(range(len(train_acc)), train_acc, 'r', label='Training accuracy')
  plt.plot(range(len(val_acc)), val_acc, 'b', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.savefig("accuracy.png")

  plt.figure()
  plt.plot(range(epochs), train_loss, 'r', label='Training loss')
  plt.plot(range(epochs), val_loss, 'b', label='Validation loss')
  plt.title('Training and validation CrossEntropy Loss')
  plt.legend()
  plt.savefig("loss.png")
