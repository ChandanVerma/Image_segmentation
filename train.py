import warnings
warnings.filterwarnings('ignore')
import os 
import sys
from tqdm import tqdm
import cv2
import albumentations as albu
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model import Unet
import numpy as np 
import torch.nn.functional as F

IMG_DIR = 'D:/HERE/data/binary_lane_bdd/Images'
MASK_DIR = 'D:/HERE/data/binary_lane_bdd/Labels'

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype("float32")

class CarDataset(Dataset):
    def __init__(self, img_folder, mask_folder, img_list = None,  transforms = None, preprocessing = True):
        self.mask_folder = mask_folder
        self.img_folder = img_folder
        self.img_ids = img_list
        self.mask_ids = img_list
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_name = self.img_ids[idx]
        img_path = os.path.join(self.img_folder, img_name)
        img = cv2.imread(img_path)
        mask_name = self.mask_ids[idx]
        mask_path = os.path.join(self.mask_folder, img_name)
        mask = cv2.imread(mask_path)
        img = np.rollaxis(img, 2)
        mask = np.rollaxis(mask, 2)
        if self.preprocessing:
            img, mask = self.transforms(image = img, masks = mask)
        
        return img, mask



def train_transformation():
    train_transform = [
        albu.HorizontalFlip(p = 0.5),
        albu.ShiftScaleRotate(
            scale_limit = 0.5,
            rotate_limit = 0,
            shift_limit = 0.1,
            p = 0.5,
            border_mode=0
        ),
        albu.GridDistortion(
            p = 0.5
        ),
        albu.Resize(320, 640),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albu.Lambda(image= to_tensor,mask=to_tensor)
    ]

    return albu.Compose(train_transform)

def valid_transformation():
    valid_transform = [
        albu.Resize(320, 640),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albu.Lambda(image= to_tensor,mask= to_tensor)
    ]
    return albu.Compose(valid_transform)

train_list, val_list = train_test_split(os.listdir(IMG_DIR), test_size = 0.1, random_state = 123)
len(train_list)
len(val_list)

train_dataset = CarDataset(img_folder = 'D:/HERE/data/binary_lane_bdd/Images', 
                           mask_folder = 'D:/HERE/data/binary_lane_bdd/Labels',
                           img_list = train_list,
                           transforms= train_transformation(),
                           preprocessing=False)

train_dataset[1]
valid_dataset = CarDataset(img_folder = 'D:/HERE/data/binary_lane_bdd/Images', 
                           mask_folder = 'D:/HERE/data/binary_lane_bdd/Labels',
                           img_list = val_list,
                           transforms= train_transformation(),
                           preprocessing=False)

len(train_dataset)
len(valid_dataset)

train_loader = DataLoader(train_dataset, num_workers= 0, batch_size= 1, shuffle= True)
val_loader = DataLoader(valid_dataset, num_workers=0, batch_size= 1, shuffle= False)

len(train_loader.dataset)
len(val_loader.dataset)

model = Unet(num_channels = 3, num_classes = 2)

from utils import BCEDiceLoss
import torch.optim as optimizer
import numpy as np  

criterion = BCEDiceLoss(eps= 1.0, activation=None)
optimizer = optimizer.Adam(model.parameters(), lr = 0.001)
current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor= 0.2, patience= 2, cooldown= 2)

n_epochs = 10
train_loss_list = []
valid_loss_list = []
dice_score_list = []
lr_rate_list = []
valid_loss_min = np.Inf

from tqdm.auto import tqdm as tq
train_on_gpu = torch.cuda.is_available()
for epoch in range(1, n_epochs + 1):

    train_loss = 0.0
    valid_loss = 0.0
    dice_score = 0.0

    model.train()
    bar = tq(train_loader, postfix= {"train_loss":0.0})
    for data, target in bar:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        data = data.float()
        target = target.float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        bar.set_postfix(ordered_dict={"train_loss":loss.item()})

    model.eval()
    del data, target
    with torch.no_grad():
        bar = tq(val_loader, postfix= {"valid_loss": 0.0, "dice_score": 0.0})
        for data, target in bar:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)
            dice_coef = dice_no_threshold(output.cpu(), target.cpu()).item()
            dice_score = dice_coef * data.size(0)
            bar.set_postfix(ordered_dict={ 'valid_loss': valid_loss, 'dice_score': dice_coef})

    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(val_loader.dataset)
    dice_score = dice_score/len(val_loader.dataset)

    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    dice_score_list.append(dice_score)
    lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups][0])

    print('Epoch: {} Training Loss: {} Validation Loss: {} Dice_loss: {}'.format(epoch, train_loss, valid_loss, dice_score))

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased {} --> {} . Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict, 'model_seg.pt')
        valid_loss_min = valid_loss

    schedular.step(valid_loss)

