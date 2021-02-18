import glob
#Using functions from workshops
from funcsForCoursework import read_file, array_to_image
#Initiate image directories
testing_image_path = 'testing/'
training_image_path = 'training/'
#Make list of file names
testing_images = glob.glob(testing_image_path + '*image*')
training_images = glob.glob(training_image_path + '*image*')
#Join training and testing images
disp_all_images = training_images + testing_images
#Display each image in list
for file in disp_all_images:
    img_array = read_file(file)
    img = array_to_image(img_array * 255)
    img.show()