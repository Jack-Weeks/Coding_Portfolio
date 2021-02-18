import numpy as np
import PIL
from demonsReg import runRegistration
from funcsForCoursework import read_file, resampImageWithDefField, calcNCC
import glob
from scipy.ndimage import gaussian_filter

def save_figs(image_array, output_dir):
    image = PIL.Image.fromarray(image_array * 255)
    image = image.convert('L')
    image.save(output_dir)
#Locate training images
training_img_folder ='training/'
#Allocate space for output
average_im = np.zeros((240,178), np.float)

imagelist = []
filenames = []
### get training images
for training_image in glob.glob(training_img_folder + '*image*'):
    imagelist.append(read_file(training_image))
    filenames.append(training_image)
#
#     ####Calculate average
# for image in imagelist:
#     original_avg = original_avg+image/len(filenames)
#
# initial_average = PIL.Image.fromarray(original_avg)

#### iterative process ####

N = 0
while N < 10: #10 iterations were given
    print('Iteration', N)

    imagelist = []
    ### get training images
    for training_image in glob.glob(training_img_folder + '*image*'):
        imagelist.append(read_file(training_image))

    if N == 9: #For last iteration we want to save deformation fields
        def_fields = []
    if N == 0: #For First iteration we want to generate an average template
        for image in imagelist:
            average_im = average_im+image/len(filenames)

    for i, image in enumerate(imagelist): #Register each image to the average

        image, field = runRegistration(image, average_im, disp_final_result=False, sigma_elastic=2, sigma_fluid=2, num_lev=3)
        imagelist[i] = image
        if N == 9:
            def_fields.append(field)

    average_im = np.zeros((240,178), np.float)

    for image in imagelist: #generate new average image from deformed images
        average_im = average_im + image / len(filenames)

    N += 1
#Output the final average template
final_average = PIL.Image.fromarray(average_im * 255)
final_average = final_average.convert('L')
final_average.save('Prior_Dir/Average_Template.png')


#Initiate prior space
grey_matter = np.zeros((240,178), np.float)
white_matter = np.zeros((240,178), np.float)
CSF = np.zeros((240, 178), np.float)
non_brain = np.zeros((240, 178), np.float)
#Get each training image
for item, training_image in enumerate(glob.glob(training_img_folder + '*image*')):
    #Rename to get each segment file
    seg1 = training_image.replace('image', 'seg1')
    seg2 = training_image.replace('image', 'seg2')
    seg3 = training_image.replace('image', 'seg3')

    seg1 = read_file(seg1)
    seg2 = read_file(seg2)
    seg3 = read_file(seg3)

    deformedseg1 = resampImageWithDefField(seg1, def_fields[item])
    deformedseg2 = resampImageWithDefField(seg2, def_fields[item])
    deformedseg3 = resampImageWithDefField(seg3, def_fields[item])

    grey_matter = (grey_matter + (deformedseg1/20))
    white_matter = (white_matter + (deformedseg2/20))
    CSF = CSF + (deformedseg3/20)

non_brain = np.nan_to_num((1 - (grey_matter + white_matter + CSF)), nan=1)

#Apply Gaussian Smoothing
grey_matter = gaussian_filter(grey_matter, sigma=4)
white_matter = gaussian_filter(white_matter, sigma=4)
CSF = gaussian_filter(CSF, sigma=4)
non_brain = gaussian_filter(non_brain, sigma=4)

grey_matter_image = PIL.Image.fromarray(grey_matter * 255)
white_matter_image = PIL.Image.fromarray(white_matter * 255)
CSF_image = PIL.Image.fromarray(CSF * 255)
non_brain_image = PIL.Image.fromarray(non_brain * 255)

save_figs(grey_matter, 'Prior_Dir/GM.png')
save_figs(white_matter, 'Prior_Dir/WM.png')
save_figs(CSF, 'Prior_Dir/CSF.png')
save_figs(non_brain, 'Prior_Dir/NonBrain.png')

grey_matter_image.show()
white_matter_image.show()
CSF_image.show()
non_brain_image.show()


