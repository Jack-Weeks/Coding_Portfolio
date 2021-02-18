import numpy as np
import PIL
import PIL as pl
import glob
from demonsReg import runRegistration
from funcsForCoursework import read_file, resampImageWithDefField, calcNCC

## convert array to image
def array_to_image(array):
    minimal_value = np.min(array)
    maximal_value = np.max(array)
    if minimal_value < 0 or maximal_value > 255:
        array = 255*(array-minimal_value)/(maximal_value-minimal_value)
    array_uint8 = array.astype('uint8')
    return pl.Image.fromarray(array_uint8, 'L') #saves as 8-bit pixels

## save array as image file
def save_file(array,filename):
    imagefile = array_to_image(array)
    imagefile.save(filename)



def ImageLNCCMap(image1, image2, window_size):
    img_dims_y, img_dims_x = image1.shape[:2]
    pad_img1 = np.pad(image1, pad_width=(window_size, window_size), mode='edge')
    pad_img2 = np.pad(image2, pad_width=(window_size, window_size), mode='edge')
    output_img = np.zeros((img_dims_y, img_dims_x), dtype='float64')

    for y in range(window_size, 240 - window_size):
        for x in range(window_size, 178 - window_size):
            img1 = pad_img1[y - window_size: y + window_size, x - window_size:x + window_size]
            img2 = pad_img2[y - window_size:y + window_size, x - window_size:x + window_size]
            LNCC = calcNCC(img1, img2)
            output_img[y - window_size, x - window_size] = (LNCC)
            #print(LNCC)

    return output_img

def LNCC(image1, image2, x,y, window_size):
    pad_img1 = np.pad(image1, pad_width=(window_size, window_size), mode='edge')
    pad_img2 = np.pad(image2, pad_width=(window_size, window_size), mode='edge')
    img1 = pad_img1[y - window_size: y + window_size, x - window_size:x + window_size]
    img2 = pad_img2[y - window_size:y + window_size, x - window_size:x + window_size]
    LNCC = (calcNCC(img1, img2))
    return LNCC





test_img_path = 'testing/'
training_img_folder ='training/'
out_dir = 'MA'

deformedsegs =[]
LNCCMaps = []
for test_img in glob.glob(test_img_path + '*image*'):
    test_img_name = test_img.replace('.png','').replace('testing\\','')
    print(test_img_name)
    test_img = read_file(test_img)

    for training_image in glob.glob(training_img_folder + '*image*'):
        print(training_image)
        seg1 = training_image.replace('image', 'seg1')
        seg2 = training_image.replace('image', 'seg2')
        seg3 = training_image.replace('image', 'seg3')


        training_image = read_file(training_image)
        seg1 = np.nan_to_num(read_file(seg1),nan=0)
        seg2 = np.nan_to_num(read_file(seg2),nan=0)
        seg3 = np.nan_to_num(read_file(seg3),nan=0)

        image, field = runRegistration(training_image, test_img, disp_final_result=False)

        deformedseg1 = resampImageWithDefField(seg1, field)
        deformedseg2 = resampImageWithDefField(seg2, field)
        deformedseg3 = resampImageWithDefField(seg3, field)
        case = [deformedseg1, deformedseg2, deformedseg3]
        deformedsegs.append(case)

        LNCC_map = ImageLNCCMap(image, test_img, window_size= 3)
        LNCCMaps.append(LNCC_map)

    stackedLNCC = np.stack(LNCCMaps, axis=2)
    maxLNCC = np.zeros((240, 178))
    output1 = np.zeros((240, 178), dtype='float64')
    output2 = np.zeros((240, 178), dtype='float64')
    output3 = np.zeros((240, 178), dtype='float64')

    for j in range(240):
        for i in range(178):
            maxLNCC = np.nan_to_num(np.amax(stackedLNCC, axis=2))
            maxreg = deformedsegs[int(maxLNCC[j, i])]
            Segmentation_LNCCs = []
            for segmentation in maxreg:
                segmentation_LNCC = np.nan_to_num(LNCC(segmentation, test_img, i, j, 3))
                Segmentation_LNCCs.append(segmentation_LNCC)

            if np.any(Segmentation_LNCCs) > 0.33:

                if Segmentation_LNCCs.index(max(Segmentation_LNCCs)) == 0 and Segmentation_LNCCs[0] > 0.33:
                    output1[j, i] = 1
                    output2[j, i] = 0
                    output3[j, i] = 0
                elif Segmentation_LNCCs.index(max(Segmentation_LNCCs)) == 1 and Segmentation_LNCCs[1] > 0.33:
                    output1[j, i] = 0
                    output2[j, i] = 1
                    output3[j, i] = 0
                elif Segmentation_LNCCs.index(max(Segmentation_LNCCs)) == 2 and Segmentation_LNCCs[2] > 0.33:
                    output1[j, i] = 0
                    output2[j, i] = 0
                    output3[j, i] = 1
                else:
                    output1[j, i] = 0
                    output2[j, i] = 0
                    output3[j, i] = 0

            else:
                output1[j, i] = 0
                output2[j, i] = 0
                output3[j, i] = 0


    save_file(output1 * 255, out_dir + '/' + test_img_name + '_MA_seg1.png')
    save_file(output2 * 255, out_dir + '/' + test_img_name + '_MA_seg2.png')
    save_file(output3 * 255, out_dir + '/' + test_img_name + '_MA_seg3.png')






