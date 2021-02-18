## Gaussian mixture models for brain tissue segmentation using expectation-maximisation
import os, sys
import numpy as np
import PIL as pl
from PIL import Image
import glob
from funcsForCoursework import resampImageWithDefField
from demonsReg import runRegistration

## read image and convert to array
def read_file(filename):
    imagefile = pl.Image.open(filename)
    image_array = np.array(imagefile.getdata(), np.uint8).reshape(imagefile.size[1], imagefile.size[0])
    return image_array.astype('float32')

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

## 2D MRF function
def Umrf(pik, k):
    G = np.ones((4, 4)) - np.diag([1, 1, 1, 1])
    Umrf = np.ndarray([np.size(pik, 0), np.size(pik, 1)])
    for indexX in range(0, np.size(pik, 0)):
        for indexY in range(0, np.size(pik, 1)):
            UmrfAtIndex = 0
            for j in range(0, np.size(pik, 2)):
                Umrfj=0
                # Umrf top
                if (indexY + 1) < np.size(pik, 1):
                    Umrfj += pik[indexX, indexY + 1, j]
                # Umrf bottom
                if (indexY - 1) > -1:
                    Umrfj += pik[indexX, indexY - 1, j]
                # Umrf right
                if (indexX + 1) < np.size(pik, 0):
                    Umrfj += pik[indexX + 1, indexY, j]
                # Umrf left
                if (indexX - 1) > -1:
                    Umrfj += pik[indexX - 1, indexY, j]
                UmrfAtIndex += Umrfj * G[k, j]
            Umrf[indexX,indexY] =UmrfAtIndex;
    return Umrf





## Select input data
## REMEMBER to change the path and filenames to match the priors created in the previous steps.
testing_dir = 'testing/'

for test_img in glob.glob(testing_dir + '*image*'): #Iterate over testing image list

    didNotConverge = 1
    iteration = 0
    maxIterations = 50
    numclass = 4
    beta = 2
    logLik = int(-10e8)
    oldLogLik = int(-10e8)
    tolerance = 100

    test_img_name = test_img.replace('.png', '').replace('testing\\', '') #Change name for suitable output
    test_img = read_file(test_img)

    print(test_img_name)
    priors_dir = "Prior_Dir/"
    GM_Prior = read_file(priors_dir + 'GM.png')
    WM_Prior = read_file(priors_dir + 'WM.png')
    CSF_Prior = read_file(priors_dir + 'CSF.png')
    Other_Prior = read_file(priors_dir + 'NonBrain.png')
    template = read_file(priors_dir + 'Average_Template.png')

    imgData, field = runRegistration(template, test_img, disp_final_result=False, sigma_fluid=3, sigma_elastic=2, num_lev=3)
    #Make deform priors, converting NaN values
    GM_Prior = np.nan_to_num(resampImageWithDefField(GM_Prior, field), nan=0)
    WM_Prior = np.nan_to_num(resampImageWithDefField(WM_Prior, field), nan=0)
    CSF_Prior = np.nan_to_num(resampImageWithDefField(CSF_Prior, field), nan=0)
    Other_Prior = np.nan_to_num(resampImageWithDefField(Other_Prior, field), nan=1)
    imgData = np.nan_to_num(imgData, nan=0)


    # Initialise mean and variances
    mean = np.random.rand(numclass,1)*256;
    var = (np.random.rand(numclass,1)*10)+200;

    # Allocate space for the posteriors
    classProb = np.ndarray([np.size(imgData,0),np.size(imgData,1),numclass])
    classProbSum = np.ndarray([np.size(imgData,0),np.size(imgData,1)])

    # Allocate space for the priors
    classPrior=np.ndarray([np.size(imgData,0),np.size(imgData,1),4])
    classPrior[:, :, 0] = GM_Prior/255
    classPrior[:, :, 1] = WM_Prior/255
    classPrior[:, :, 2] = CSF_Prior/255
    classPrior[:, :, 3] = Other_Prior/255

    for classIndex in range(0, numclass):
        pik = classPrior[:, :, classIndex]
        mean[classIndex] = np.sum(pik * imgData) / np.sum(pik)
        var[classIndex] = np.sum(pik * ((imgData - mean[classIndex]) ** 2)) / np.sum(pik)

    # Define MRF array to match image dimensions
    MRF=np.ndarray([np.size(imgData,0),np.size(imgData,1),numclass])
    MRF[:,:,:]=1

    ## Run EM
    # Iterative process
    while didNotConverge:
        iteration=iteration+1

        # Expectation
        classProbSum[:, :] = 0;
        for classIndex in range(0, numclass):
            gaussPdf = (1/np.sqrt(var[classIndex]*2*np.pi))*np.exp(-0.5* (imgData-mean[classIndex])**2/var[classIndex])
            classProb[:, :, classIndex] = (gaussPdf+0.0000001) * classPrior[:, :, classIndex] * MRF[:,:,classIndex]
            classProbSum[:, :] = classProbSum[:, :]+classProb[:, :, classIndex]

        # Normalise posterior
        for classIndex in range(0, numclass):
            classProb[:, :, classIndex] = classProb[:, :, classIndex] /classProbSum[:, :]

        # Maximization
        print("Iteration #"+str(iteration))
        for classIndex in range(0, numclass):
            pik = classProb[:, :, classIndex]
            mean[classIndex] = np.sum(pik*imgData) / np.sum(pik)
            var[classIndex] = np.sum(pik*((imgData - mean[classIndex])**2)) / np.sum(pik)
            MRF[:,:,classIndex] = np.exp(-beta * Umrf(classProb, classIndex))
            print("Class index "+ str(classIndex) + ": mean = "+str(mean[classIndex]) + ", variance = " + str(var[classIndex]))

        # Calculate log likelihood
        oldLogLik = logLik
        logLik = np.sum(np.log(classProbSum))
        diffLogLik = oldLogLik - logLik
        print("Loglikelihood change = " + str(diffLogLik))

        if abs(diffLogLik)<=tolerance:
          didNotConverge=0
          print("Converged after " + str(iteration) + " iterations")

        if iteration>=maxIterations:
            didNotConverge=0
            print("Reached "+ str(maxIterations) + " iterations (the maximum) and did not coverge")


    ## Save segmented images as separate PNG files
    for classIndex in range(0, numclass):
        print("Saving "+"seg"+str(classIndex)+".png")
        save_file(classProb[ : ,: ,classIndex] * 255, "GMM/" + test_img_name + '_GMM_seg' + str(classIndex + 1) + ".png")

