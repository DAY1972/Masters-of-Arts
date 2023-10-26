import tensorflow as tf
import pandas as pd
import numpy as np
import pathlib
import PIL
from PIL import Image
import os
import matplotlib.pyplot as plt

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

loaded_model = tf.keras.models.load_model('VGG16_trainable_True_299_299.keras')

test_dir = pathlib.Path('M:/Хакатоны/ML Art challenge/test/')
path_to_files = list(test_dir.glob('*'))
image_name = []
name = []
for path in path_to_files:
    name.append(list(str(path).split('\\'))[-1].split('.')[-2])
    image_name.append(list(str(path).split('\\'))[-1])
df = pd.DataFrame.from_dict({'image_name':image_name, 'name':name})
df.set_index('name', inplace=True)

test_resized_dir = pathlib.Path('M:/Хакатоны/ML Art challenge/test_resized_299_299/')
path_to_resized_files = list(test_resized_dir.glob('*.png'))

predict_arr = np.zeros((len(path_to_resized_files), 40))
image_name = []
label_id = []
i = 0
for path in path_to_resized_files:
    file_name = list(str(path).split('\\'))[-1].split('.')[-2]
    image_name.append(df.loc[file_name][0])
    image = tf.keras.preprocessing.image.load_img(
        path, grayscale=False, color_mode="rgb", target_size=None,
        interpolation="nearest")
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = loaded_model.predict(input_arr)
    predict_arr[i] += predictions[0]
    i += 1

predict = np.argmax(predict_arr, axis=1)
tree = list(os.walk('M://Хакатоны//ML Art challenge//train_resized_299_299//'))[0][1]
tree = [int(x) for x in tree]
for i in predict:
    label_id.append(tree[i])
df_predict = pd.DataFrame.from_dict({'image_name':image_name, 'label_id':label_id})
df_predict.to_csv(path_or_buf='M:/Хакатоны/ML Art challenge/prediction.csv', sep='\t')
'''
for path in path_to_files:
  file = str(path).split('\\')[-1].split('.')[-2]
  label.append(train_labels['labels'].loc[file])
'''

