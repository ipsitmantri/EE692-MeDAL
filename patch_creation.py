import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image
parent = 'dataset_TMA'
path_images = os.path.join(parent,'TMA_images')
train = ['ZT111_4_A','ZT111_4_B','ZT111_4_C','ZT199_1_A','ZT199_1_B','ZT204_6_A','ZT204_6_B']
for t in train:
    train_path = os.path.join(path_images,t)
    for image in os.listdir(train_path):
        img = Image.open(os.path.join(train_path,image))
        