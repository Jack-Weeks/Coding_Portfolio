import matplotlib.pyplot as plt
import numpy as np
import PIL as pl
import PIL as PIL
from pathlib import Path
from PIL import Image
# from funcsForCoursework import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import skimage.io
import time
from funcsForCoursework import read_file, calcNCC
from skimage.exposure import rescale_intensity

from skimage.io import imread
train_dr = Path('training/')



image1 = read_file('training/STORM_04_v2_seg1.png')
image1show = PIL.Image.fromarray(image1)
image2 = read_file('training/STORM_02_v1_seg2.png')
image2show= PIL.Image.fromarray(image2)

window_size = 3
img_dims_y, img_dims_x = image1.shape[:2]
pad_img1 = np.pad(image1,pad_width=(window_size, window_size), mode='edge')
pad_img2 = np.pad(image2,pad_width=(window_size, window_size),mode='edge')
output_img = np.zeros((img_dims_y, img_dims_x), dtype='float64')


for y in range(window_size, 240 - window_size):
    for x in range(window_size, 178 - window_size):
        img1 = pad_img1[y - window_size: y + window_size, x-window_size:x + window_size]
        img2 = pad_img2[y-window_size:y + window_size, x-window_size:x + window_size]
        LNCC = calcNCC(img1, img2)
        output_img[y-window_size, x-window_size] = abs(LNCC * 255)
        print(LNCC)
        # if np.isnan(LNCC):
        #     output_img[i-window_size,j-window_size] = 1
# output_img = rescale_intensity(output_img, in_range=(0, 255))
# output_img = (output_img * 255).astype("uint8")
output = PIL.Image.fromarray(output_img)
output.show()
image1show.show()
image2show.show()

def LNCC(image1, image2, window_size):
    img_dims_y, img_dims_x = image1.shape[:2]
    pad_img1 = np.pad(image1, pad_width=(window_size, window_size), mode='edge')
    pad_img2 = np.pad(image2, pad_width=(window_size, window_size), mode='edge')
    output_img = np.zeros((img_dims_y, img_dims_x), dtype='float64')

    for y in range(window_size, 240 - window_size):
        for x in range(window_size, 178 - window_size):
            img1 = pad_img1[y - window_size: y + window_size, x - window_size:x + window_size]
            img2 = pad_img2[y - window_size:y + window_size, x - window_size:x + window_size]
            LNCC = calcNCC(img1, img2)
            output_img[y - window_size, x - window_size] = abs(LNCC * 255)
            print(LNCC)


    return output_img