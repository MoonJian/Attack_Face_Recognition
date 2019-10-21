from tqdm import tqdm
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import sys
import os
import pandas as pd
import cv2
# sys.path.append('G:/code/Face_Adversarial/face_evo/align')
sys.path.insert(0, '.')
from backbone.model_irse import IR_50
import json

def json_load(path):
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config

config = json_load('./Name.json')

def img2tensor(img):    
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    img = np.reshape(img, [1, 3, 112, 112])
    img = np.array(img, dtype = np.float32)
    img = (img - 127.5) / 128.0
    img = torch.from_numpy(img)
    return img

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

model1 = IR_50([112,112])
model1.load_state_dict(torch.load('./IR_50.pth',map_location='cpu'))
model1.eval()

dev = pd.read_csv('../data/securityAI_round1_dev.csv')
dev_image = dev.ImageName.values
dev_person = dev.PersonName.values
dic_name_r = dict(zip(dev_person,dev_image))

mean_feat1 = pd.read_csv('./mean.csv').rename(columns = {'Unnamed: 0':'name'})

path1="../data/securityAI_round1_images/"

for person in config.items():

    steps = person[1]['steps']
    coeff = person[1]['coeff']
    bad_name = person[1]['name']
    
    print('bad_name length:',len(bad_name))
    print('steps:',steps)
    print('coeff:',coeff)
    
    alpha = coeff/steps
    info = []
    for nnn in tqdm(bad_name):

        img = np.array(Image.open(path1+nnn))

        img_r_raw = img2tensor(img)

        img_r = img2tensor(img)


        raw1 = l2_norm(model1(img_r))

        img_r1_norm = l2_norm(model1(img_r))

        N = steps

        arg1 = mean_feat1.iloc[:,1:].dot(img_r1_norm.detach().numpy().T)

        img_self1_norm = mean_feat1.iloc[arg1.sort_values(by = [0],ascending = False).index[0],1:].values.reshape(1,512).astype('float32')

        img_self1_norm = torch.from_numpy(img_self1_norm)

        for x in range(N):

            print('Attack:'+str(x+1))

            img_r.requires_grad = True

            img_r_norm = l2_norm(model1(img_r)) 

            loss = F.mse_loss(img_r_norm, img_self1_norm)

            print('Loss:',loss)
            model1.zero_grad()

            loss.backward(retain_graph=True)

            data_grad = img_r.grad.data


            norm_data_grad = torch.norm(data_grad, p=2, dim=2,keepdim = True)
            norm_data_grad = torch.norm(norm_data_grad, p=2, dim=3,keepdim = True)
            sign_data_grad = data_grad/norm_data_grad
            img_r = img_r + alpha * sign_data_grad

            img_r = torch.clamp(img_r,-127.5/128.0,127.5/128.0)

            img_r = torch.from_numpy(img_r.detach().numpy())

        gc.collect()
        S1 = raw1.detach().numpy().dot(img_r_norm.detach().numpy().T)
        print('Similarity with raw1:',S1)

        info.append([nnn,S1[0][0],x+1])
        fake = (img_r.squeeze().detach().cpu().numpy().swapaxes(1, 0).swapaxes(2, 1)*128.0+127.5)
        test_path = '../data/images/'
        os.makedirs(test_path,exist_ok=True)
        cv2.imwrite(test_path+nnn,fake[...,::-1],[int(cv2.IMWRITE_JPEG_QUALITY),96])
    