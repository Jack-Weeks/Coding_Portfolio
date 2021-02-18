import numpy as np
import PIL
import PIL as pl
import glob
from demonsReg import runRegistration
from funcsForCoursework import read_file, resampImageWithDefField, calcNCC, save_file

##Question 4b
def ImageLNCCMap(image1, image2, window_size):
    img_dims_y, img_dims_x = image1.shape[:2]
    pad_img1 = np.pad(image1, pad_width=(window_size, window_size), mode='edge') #Pad each image with edge values
    pad_img2 = np.pad(image2, pad_width=(window_size, window_size), mode='edge') #Pad each image with edge values
    output_img = np.zeros((img_dims_y, img_dims_x), dtype='float64') #Make output image

    for y in range(window_size, img_dims_y - window_size): #Iterate over y coordinates
        for x in range(window_size, img_dims_x - window_size): #Iterate over x coordinates
            img1 = pad_img1[y - window_size: y + window_size, x - window_size:x + window_size]
            img2 = pad_img2[y - window_size:y + window_size, x - window_size:x + window_size]
            LNCC = calcNCC(img1, img2) #Calculate LNCC
            output_img[y - window_size, x - window_size] = (LNCC) #Update output image with LNCC value

    return output_img

# def LNCC(image1, image2, x,y, window_size):
#     pad_img1 = np.pad(image1, pad_width=(window_size, window_size), mode='edge')
#     pad_img2 = np.pad(image2, pad_width=(window_size, window_size), mode='edge')
#     img1 = pad_img1[y - window_size: y + window_size, x - window_size:x + window_size]
#     img2 = pad_img2[y - window_size:y + window_size, x - window_size:x + window_size]
#     LNCC = (calcNCC(img1, img2))
#     return LNCC


test_img_path = 'testing/'
training_img_folder ='training/'
out_dir = 'MA'

deformedsegs =[]
LNCCMaps = []


for test_img in glob.glob(test_img_path + '*image*'):
    test_img_name = test_img.replace('.png', '').replace('testing\\', '') #Make output filename appropriate
    test_img = read_file(test_img)
    for training_image in glob.glob(training_img_folder + '*image*'):
        #Change name to get filename of each segment
        seg1 = training_image.replace('image', 'seg1')
        seg2 = training_image.replace('image', 'seg2')
        seg3 = training_image.replace('image', 'seg3')


        training_image = read_file(training_image)
        seg1 = read_file(seg1)
        seg2 = read_file(seg2)
        seg3 = read_file(seg3)

        image, field = runRegistration(training_image, test_img, disp_final_result=False, sigma_elastic=1, sigma_fluid=1, num_lev= 7)

        deformedseg1 = np.nan_to_num(resampImageWithDefField(seg1, field), nan=0)
        deformedseg2 = np.nan_to_num(resampImageWithDefField(seg2, field), nan=0)
        deformedseg3 = np.nan_to_num(resampImageWithDefField(seg3, field), nan=0)
        case = [deformedseg1, deformedseg2, deformedseg3]
        deformedsegs.append(case)

        LNCC_map = ImageLNCCMap(image, test_img, window_size=3)
        LNCCMaps.append(LNCC_map)

    stackedLNCC = np.stack(LNCCMaps, axis=2) #Stack all LNCC maps into a 3d Matrix
    maxLNCC = np.zeros((240, 178))
    output1 = np.zeros((240, 178), dtype='float64')
    output2 = np.zeros((240, 178), dtype='float64')
    output3 = np.zeros((240, 178), dtype='float64')

    for j in range(240):
        for i in range(178):
            maxLNCC[j,i] = int(np.argmax(stackedLNCC[j,i,:], axis=0)) #Make image, with each pixel equal to the case with the highest registration case
            #By looking along the 3rd axis for each pixel for the largest value
            maxreg = deformedsegs[int(maxLNCC[j, i])] # Then we get the deformed images of the maximum registration at the pixel
            deformed_pix = [maxreg[0][j,i], maxreg[1][j,i],maxreg[2][j,i]] #Create list of [Seg1,Seg2,Seg3]
            max_seg_value = max(deformed_pix) #Find the maximum pixel value of all segmentations
            max_seg_idx = deformed_pix.index(max_seg_value) #Get the index of the maximum value 0 = seg1, 1=seg2 2= seg3

            if max_seg_value > 0.33: #check if greater than random
                if max_seg_idx == 0: #If the maximum value comes from seg1
                    output1[j, i] = 1
                    output2[j, i] = 0
                    output3[j, i] = 0
                if max_seg_idx == 1: #If Maximum value comes from seg2
                    output1[j, i] = 0
                    output2[j, i] = 1
                    output3[j, i] = 0
                if max_seg_idx == 2: #If maxumum value comes from seg3
                    output1[j, i] = 0
                    output2[j, i] = 0
                    output3[j, i] = 1
            else: #If not significant
                output1[j, i] = 0
                output2[j, i] = 0
                output3[j, i] = 0
#Save and output.
    save_file(output1 * 255, out_dir + '/' + test_img_name + '_MA_seg1.png')
    save_file(output2 * 255, out_dir + '/' + test_img_name + '_MA_seg2.png')
    save_file(output3 * 255, out_dir + '/' + test_img_name + '_MA_seg3.png')
