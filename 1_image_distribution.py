import os
import pandas as pd
import pathlib

for folder_name in range(40):
    folder = 'M:\\Хакатоны\\ML Art challenge\\train_resized_299_299\\' + str(folder_name)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
train_labels = pd.read_csv('train.csv', sep='\t')
#train_labels['labels'] = train_labels['label_id'].map(one_hot_encoding)
train_labels['only_image_name'] = train_labels['image_name'].str.partition(sep='.')[0] + '.png'
train_labels.drop(['image_name'], axis=1, inplace=True)
train_labels.set_index(keys='only_image_name', inplace=True)
train_labels.sort_index(inplace=True)

data_dir = pathlib.Path('M:/Хакатоны/ML Art challenge/train_resized_299_299/')
path_to_files = list(data_dir.glob('*.png'))

file_source_ground = 'M:/Хакатоны/ML Art challenge/train_resized_299_299/'
file_destination_ground = 'M:/Хакатоны/ML Art challenge/train_resized_299_299/'
for file in path_to_files:
    f = str(file).split('\\')[-1]
    file_source = file_source_ground + f
    file_destination = file_destination_ground + str(train_labels.loc[f][-1]) + '/' + f
    os.replace(file_source, file_destination)
