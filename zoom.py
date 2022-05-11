#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision.models as model
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
import os
import glob
from tqdm import tqdm
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import timm.models as tmodels

parser = argparse.ArgumentParser()

parser.add_argument('-gpu')
parser.add_argument('-train')
parser.add_argument('-val')
parser.add_argument('-test')
parser.add_argument('-save')
parser.add_argument('-model')
parser.add_argument('-bsize', default=64)

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu




# In[2]:


random_seed = 1120
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


# In[3]:


import matplotlib.image as img
import matplotlib.pyplot as plt

def imshow(x):
    x = img.imread(x)
    plt.imshow(x)
    plt.show()


# In[4]:

zoom_path = '/media/data2/rjsdn/zoom/data/dataset/'
D_path = '/media/data1/dataset/DFDC/'
P_path = '/media/data1/dataset/PGGAN/256x256/'

SAVE_PATH='weights/'+args.save
os.makedirs(SAVE_PATH,exist_ok=True)


train_path = args.train
val_path = args.val
test_path = args.test

w,h=224,224

transform = transforms.Compose([
    transforms.Resize((w,h)),
    transforms.ToTensor()
])

train_img = datasets.ImageFolder(train_path,transform=transform)
val_img = datasets.ImageFolder(val_path,transform=transform)
test_img = datasets.ImageFolder(test_path,transform=transform)

bsize=int(args.bsize)

train_loader = DataLoader(train_img,batch_size=bsize,shuffle=True,num_workers=8)
val_loader = DataLoader(val_img,batch_size=bsize,shuffle=True,num_workers=8)
test_loader = DataLoader(test_img,batch_size=bsize,shuffle=False,num_workers=8)






# In[5]:

if args.model=='X':
    Model=tmodels.xception()
    testmodel=tmodels.xception()
if args.model=='R18':
    Model=tmodels.resnet18()
    testmodel=tmodels.resnet18()

# Model=tmodels.resnet18(pretrained=False)
num_ftrs = Model.fc.in_features
Model.fc = nn.Linear(num_ftrs,2)
    
device = torch.device("cuda:0")
Model=Model.to(device)

opt = torch.optim.Adam(Model.parameters(), lr=0.001,weight_decay=1e-1)
criterion = nn.CrossEntropyLoss()


# In[6]:


EPOCH=30



# In[7]:


history={
    'train_acc':[],
    'val_acc':[],
    'train_loss':[],
    'val_loss':[],
}

best_loss = np.inf
best_acc=0
for epo in range(EPOCH):
    train_loss=0
    train_acc=0
    Model.train()
    count=0
    for x, y in tqdm(train_loader):
        opt.zero_grad()
        x=x.to(device)
        y=y.to(device)
        output = Model(x)
        loss = criterion(output,y)
        _, index = torch.max(output,1)
        train_acc += (index==y).sum().float().item() / len(output)
        train_loss += float(loss.item())
        loss.backward(retain_graph=True)
        opt.step()
        count+=1
    
    train_loss /= count
    train_acc /= count

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    
    val_loss=0
    val_acc=0    
    Model.eval()
    with torch.no_grad():
        count=0
        for x,y in tqdm(val_loader):
            x=x.cuda()
            y=y.cuda()
            output=Model(x)
            loss = criterion(output,y)
            _, index = torch.max(output,1)
            val_acc += (index==y).sum().float().item() / len(output)
            val_loss += float(loss.item())
            count+=1
        val_acc/=count
        val_loss/=count
    
    print(f'[{epo+1}/{EPOCH}] TL={train_loss: .8f} TA={train_acc*100:.3f} || VL={val_loss:.8f} VA={val_acc*100:.2f}')
    
    
    if best_loss > val_loss:
        print('best weights changed!!!')
        best_acc=val_acc
        best_loss=val_loss
        torch.save(Model.state_dict(),SAVE_PATH+'/best.pth')
        
    
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
#    if train_loss < 0.01: break
    
print("\n\nTrain Done")


# In[14]:


# testmodel=model.resnet18(pretrained=False)
num_ftrs = testmodel.fc.in_features
testmodel.fc = nn.Linear(num_ftrs,2)
testmodel.load_state_dict(torch.load(SAVE_PATH+'/best.pth'))
testmodel.cuda()


# In[15]:


from sklearn.metrics import confusion_matrix

test_loss=0
acc = 0
count=0
p,r=torch.tensor([0]),torch.tensor([0])
testmodel.eval()
with torch.no_grad():
    for x,y in tqdm(test_loader):
        count+=1
        x=x.to(device)
        y=y.to(device)
        output = testmodel(x)
        pred=output.argmax(dim=1)
        p=torch.cat([p,pred.to("cpu")])
        r=torch.cat([r,y.to("cpu")])
        acc += pred.eq(y.view_as(pred)).sum().item() / len(output)
    acc /= count
    print("ACC : %.2f"%(acc*100))


# In[9]:


import pandas as pd

pd.DataFrame(history).to_csv(SAVE_PATH+'/res.csv', index=False)


# In[10]:


CM = confusion_matrix(r,p)
print(CM)


# In[ ]:




