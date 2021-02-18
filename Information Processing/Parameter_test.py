from demonsReg import runRegistration
from skimage.io import imread
import numpy as np
import PIL as pl
from funcsForCoursework import array_to_image
def read_file(filename):
    imagefile = pl.Image.open(filename)
    image_array = np.array(imagefile.getdata(), np.uint64).reshape(imagefile.size[1], imagefile.size[0])
    return image_array.astype('double')

image1 = read_file('training/STORM_04_v1_image.png')
image2 = read_file('training/STORM_06_v2_image.png')
image3 = read_file('training/STORM_10_v2_image.png')

img1, field1 = runRegistration(image3, image1, sigma_elastic=2, sigma_fluid=3, num_lev=3)
img2, field2 = runRegistration(image2, image1, sigma_elastic=2, sigma_fluid=3, num_lev=3)
img3, field3 = runRegistration(image3, image1, sigma_elastic=2, sigma_fluid=3, num_lev=3)
img4, field4 = runRegistration(image2, image1, sigma_elastic=2, sigma_fluid=3, num_lev=3)

img1 = array_to_image(img1)
img2 = array_to_image(img2)
img3 = array_to_image(img3)
img4 = array_to_image(img4)

img1.show()
img2.show()
img3.show()
img4.show()