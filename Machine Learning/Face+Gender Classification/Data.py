import pandas as pd
import os
import glob
import numpy as np
from tqdm import tqdm
import shutil
import pickle





data = pd.read_csv('labels.csv', delim_whitespace=True)
data.set_index('img_name', inplace=True)
data.replace(-1,0, inplace= True)
data.to_pickle('C:/Users/weeks/Gina/data_pkl.pkl')


for i in ['train', 'valid']:
    os.mkdir(os.path.join('C:/Users/weeks/Gina/', i))


filenames = glob.glob('C:/Users/weeks/Gina/img/*jpg')
shuffle = np.random.permutation(len(filenames))

training_dict = {}
training_filenames = []

for j in tqdm(shuffle[:4500]):
    file = filenames[j].split('\\')[-1]
    labels = np.array(data[data.index == file])
    training_dict[file] = labels
    training_filenames.append(file)
    shutil.copy('C:/Users/weeks/Gina/img/' + file, 'C:/Users/weeks/Gina/train/' + file)
#
valid_dict = {}
valid_filenames = []

for j in tqdm(shuffle[4500:]):
    file = filenames[j].split('\\')[-1]
    labels = np.array(data[data.index == file])
    valid_dict[file] = labels
    valid_filenames.append(file)
    shutil.copy('C:/Users/weeks/Gina/img/'+ file, 'C:/Users/weeks/Gina/valid/' + file)


training_dataframe = pd.DataFrame(training_dict.items())
training_dataframe.columns = ['id', 'labels']
training_dataframe.set_index([training_dataframe['id']], inplace=True)
training_dataframe= training_dataframe.drop(columns=['id'])

training_dataframe.to_csv('C:/Users/weeks/Gina/train.csv')
training_dataframe.to_pickle('C:/Users/weeks/Gina/train.pkl')

valid_dataframe = pd.DataFrame(valid_dict.items())
valid_dataframe.columns = ['id', 'labels']
valid_dataframe.set_index([valid_dataframe['id']], inplace=True)
valid_dataframe= valid_dataframe.drop(columns=['id'])
valid_dataframe.to_csv('C:/Users/weeks/Gina/valid.csv')
valid_dataframe.to_pickle('C:/Users/weeks/Gina/valid.pkl')





