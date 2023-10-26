import pathlib
import PIL
from PIL import Image


data_dir = pathlib.Path('M:/Хакатоны/ML Art challenge/train/')
path_to_files = list(data_dir.glob('*'))

# resizing images
for path in path_to_files:
    with Image.open(path) as img:
        if img.mode == 'CMYK':
            img_rgb = img.convert('RGB')
        else:
            img_rgb = img
        path_to_save = 'M:/Хакатоны/ML Art challenge/train_resized/' + str(path).split('\\')[-1].split('.')[-2] + '.png'
        new_image = img_rgb.resize(size=(299, 299), resample=PIL.Image.LANCZOS)
        new_image.save(path_to_save)

data_dir = pathlib.Path('M:/Хакатоны/ML Art challenge/test/')
path_to_files = list(data_dir.glob('*'))


for path in path_to_files:
    with Image.open(path) as img:
        if img.mode == 'CMYK':
            img_rgb = img.convert('RGB')
        else:
            img_rgb = img
        path_to_save = 'M:/Хакатоны/ML Art challenge/test_resized/' + str(path).split('\\')[-1].split('.')[-2] + '.png'
        new_image = img_rgb.resize(size=(299, 299), resample=PIL.Image.LANCZOS)
        new_image.save(path_to_save)
