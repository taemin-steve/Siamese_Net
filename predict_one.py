import pandas as pd 
import cv2 as cv


import matplotlib.pyplot as plt

import torchvision
import torchvision.utils
import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import timm

# define BaseModel
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        self.classifier = nn.Linear(1000, 50)
        self.dropout = nn.Dropout(0.1)
        self.ReLU = nn.ReLU(inplace=False)
        
    def forward(self, x, y):
        x = self.backbone(x)
        # x = self.dropout(x)
        x = self.classifier(x)
        # x = self.ReLU(x)
        
        y = self.backbone(y)
        # y = self.dropout(y)
        y = self.classifier(y)
        # y= self.ReLU(y)
        
        z = F.pairwise_distance(x, y, keepdim = True)
        
        return z

# define inference

def inference(model, review_img_path, product_img_path, transform,  device):
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        review_img = cv.imread(review_img_path)
        product_img = cv.imread(product_img_path)
        
        review_img = transform(image = review_img)['image']
        product_img = transform(image = product_img)['image']
        
        review_img = review_img.float().to(device)
        product_img = product_img.float().to(device)
        
        pred = model(review_img, product_img)
    return pred

#-----------------------------------------------------------
transform = A.Compose([A.Resize(224,224),ToTensorV2()])

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

model = BaseModel()
model.load_state_dict(torch.load('./distance_EffNetBase_E_Contra.pt'))
model.eval()

review_img_path = '/home/visualinformaticslab/Siamese_Net/masked_data/product_img/0.jpg'
product_img_path = '/home/visualinformaticslab/Siamese_Net/masked_data/review_img/2_review_img/O/22.jpg'

pred = inference(model, review_img_path, product_img_path,transform, device)

print(pred)