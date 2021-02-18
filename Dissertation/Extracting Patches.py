import numpy as np
import nibabel as nib
import os

organs = ['CanalPRV', 'Heart', 'LLung', 'RLung', 'Oesophagus', 'SpinalCanal', 'Lungs', 'Spinal Canal']
patient_num = []
for x in range(50):
    patient_num.append(str(x).zfill(2))
patient_num.remove('00')


def compare_patches(cor_filepath, er_filepath, dif_filepath, x, y, z,cor_path,er_path):

    image_dif = nib.load(dif_filepath)
    image_er = nib.load(er_filepath)
    image_cor = nib.load(cor_filepath)

    imgdata_dif = np.asarray(image_dif.get_fdata())
    imgdata_er = np.asarray(image_er.get_fdata())
    imgdata_cor = np.asarray(image_cor.get_fdata())

    img_dif_datapad = np.pad(imgdata_dif, ((528 - imgdata_dif.shape[0], 0), (528 - imgdata_dif.shape[1], 0),
                                           (528 - imgdata_dif.shape[2], 0)), constant_values=0)
    img_er_datapad = np.pad(imgdata_er, ((528 - imgdata_er.shape[0], 0), (528 - imgdata_er.shape[1], 0),
                                         (528 - imgdata_er.shape[2], 0)), constant_values=0)
    img_cor_datapad = np.pad(imgdata_cor, ((528 - imgdata_cor.shape[0], 0), (528 - imgdata_cor.shape[1], 0),
                                           (528 - imgdata_cor.shape[2], 0)), constant_values=0)
    lenx = int(528 / x)
    leny = int(528 / y)
    lenz = int(528 / z)

    er_array = np.zeros((1, x, y, z))
    cor_array = np.zeros((1, x, y, z))

    er_num = 0
    cor_num = 0


    cor_path = str(cor_path)
    er_path = str(er_path)
    for i in range(0, lenx):
        for j in range(0, leny):
            for k in range(0, lenz):
                temp_patch_er = img_er_datapad[i * x:(i + 1) * x, j * y:(j + 1) * y,
                                k * z:(k + 1) * z]
                # temp_patch_dif = img_dif_datapad[i * x:(i + 1) * x, j * y:(j + 1) * y,
                #                  k * z:(k + 1) * z]
                temp_patch_cor = img_cor_datapad[i * x:(i + 1) * x, j * y:(j + 1) * y,
                                k * z:(k + 1) * z]

                if np.count_nonzero(temp_patch_er - temp_patch_cor):

                    # er_array = np.vstack((er_array, np.expand_dims(temp_patch_er, axis=0)))
                    er_nii = nib.Nifti1Image(temp_patch_er, image_er.affine, image_er.header)
                    #nib.save(er_nii, er_path)
                    np.save(er_path + str(er_num), temp_patch_er)
                    er_num += 1

                else:
                    if np.count_nonzero(temp_patch_cor):


                            #cor_array = np.vstack((cor_array, np.expand_dims(temp_patch_cor, axis=0)))
                        #cor_nii = nib.Nifti1Image(temp_patch_cor, image_cor.affine, image_cor.header)
                        #nib.save(cor_nii, cor_path)

                        np.save(cor_path + str(cor_num), temp_patch_cor)
                        cor_num += 1




    return er_array, cor_array




#

# def generate_patches(img,x,y,z):
#     img = nib.load(img)
#     imgdata = np.asarray(img.get_fdata())
#     padimage = np.pad(imgdata, ((528 - imgdata.shape[0], 0), (528 - imgdata.shape[1], 0),
#                                            (240 - imgdata.shape[2], 0)), constant_values=0)
#     lenx = int(528 / x)
#     leny = int(528 / y)
#     lenz = int(240 / z)
#
#     out_array = np.zeros((1,x,y,z))
#
#     for i in range(0, lenx):
#         for j in range(0, leny):
#             for k in range(0, lenz):
#                 temp_patch = padimage[i * x:(i + 1) * x, j * y:(j + 1) * y,
#                                 k * z:(k + 1) * z]
#                 # if np.count_nonzero(temp_patch):
    #                 out_array = np.vstack((out_array,np.expand_dims(temp_patch, axis= 0)))
#
#     return out_array

#generate_patches(er_filepath,24,24,24)


for number in range(0,50):
    sourcedir = 'NewNift/UCL_LUNG_' + patient_num[number]
    dif_dir = 'NewNift/Differences/' + patient_num[number]


    for x in organs:
        if number <= 40:
            destdir_cor = 'NewNift/Patches/train24/correct/' + str(number) +'_'
            destdir_er = 'ewNift/Patches/train24/error/' + str(number) +'_'
            if not os.path.isdir(destdir_er):
                os.makedirs(destdir_er)
            if not os.path.isdir(destdir_cor):
                os.makedirs(destdir_cor)
        if number > 40:
            destdir_cor = 'NewNift/CV/valid24/correct/' + str(number)+ '_'
            destdir_er = 'NewNift/CV/valid24/error/' + str(number)+ '_'
            if not os.path.isdir(destdir_er):
                os.makedirs(destdir_er)
            if not os.path.isdir(destdir_cor):
                os.makedirs(destdir_cor)
        cor_filepath = sourcedir + '/' + x + 'Cor_' + patient_num[number] + '.nii'
        er_filepath = sourcedir + '/' + x + 'Er_' + patient_num[number] + '.nii'
        dif_filepath = dif_dir + '/' + x + '_Diff_' + patient_num[number] + '.nii'
        if os.path.isfile(cor_filepath) and os.path.isfile(er_filepath) and os.path.isfile(dif_filepath):

            y = compare_patches(cor_filepath, er_filepath, dif_filepath, 24, 24, 24, destdir_cor, destdir_er)
            print(len(y[0]))







