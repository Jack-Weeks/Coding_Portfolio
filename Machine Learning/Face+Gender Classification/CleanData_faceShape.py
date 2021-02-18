
import torch
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from collections import Counter
from tqdm import tqdm
import PIL
from PIL import Image
import pandas as pd
import glob

import shutil
Data = pd.read_csv('/labelstoonz.csv', delim_whitespace= True)
Data.set_index('file_name', inplace=True)

PATH = '/cartoon_set/img/'
#Save_PATH = 'D:\Programming\Gina\cartoon_set'
filenames = glob.glob(PATH + '*png')
print(len(filenames))

#Sunglass_pth = 'D:/Programming/Gina/Sunglasses/'

# obscured_eyes = glob.glob(Sunglass_pth + '*png')
#
# for file in obscured_eyes:
#     file = file.split('\\')[-1]
#     Data.loc[file]['eye_color'] = 5





for i in ['train', 'valid']:
  for j in ['0','1','2','3','4']:
    if not os.path.exists(os.path.join(PATH, i, j)):
      os.makedirs(os.path.join(PATH, i, j))

i = 0
j = 0
for j in tqdm(filenames[:9500]):
    file = j.split('\\')[-1]
    # print(file)
    if Data[Data.index == file]['face_shape'].item() == 0:
        shutil.copy(PATH + file, PATH + 'train/' + '0/' + file)

    elif Data[Data.index == file]['face_shape'].item() == 1:
        shutil.copy(PATH + file, PATH + 'train/' + '1/' + file)

    elif Data[Data.index == file]['face_shape'].item() == 2:
        shutil.copy(PATH + file, PATH + 'train/' + '2/' + file)

    elif Data[Data.index == file]['face_shape'].item() == 3:
        shutil.copy(PATH + file, PATH + 'train/' + '3/' + file)

    elif Data[Data.index == file]['face_shape'].item() == 4:
        shutil.copy(PATH + file, PATH + 'train/' + '4/' + file)

    # elif Data[Data.index == file]['eye_color'].item() == 5:
    #     shutil.copy(PATH + file, Save_PATH + 'train/' + 'Obscured/' + file)

for i in tqdm(filenames[9500:]):
    file = i.split('\\')[-1]
    # print(file)
    if Data[Data.index == file]['face_shape'].item() == 0:
        shutil.copy(PATH + file, PATH + 'valid/' + '0/' + file)

    elif Data[Data.index == file]['face_shape'].item() == 1:
        shutil.copy(PATH + file, PATH + 'valid/' + '1/' + file)

    elif Data[Data.index == file]['face_shape'].item() == 2:
        shutil.copy(PATH + file, PATH + 'valid/' + '2/' + file)

    elif Data[Data.index == file]['face_shape'].item() == 3:
        shutil.copy(PATH + file, PATH + 'valid/' + '3/' + file)

    elif Data[Data.index == file]['face_shape'].item() == 4:
        shutil.copy(PATH + file, PATH + 'valid/' + '4/' + file)

    # elif Data[Data.index == file]['eye_color'].item() == 5:
    #     shutil.copy(PATH + file, Save_PATH + 'valid/' + 'Obscured/' + file)




