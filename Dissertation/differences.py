import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects, generate_binary_structure, iterate_structure, binary_dilation, binary_erosion
# img1 = nib.load('UCL_LUNG_01/HeartCor_01.nii')
# img2 = nib.load('UCL_LUNG_01/HeartEr_01.nii')
patient_num = []
for x in range(51):
    patient_num.append(str(x).zfill(2))
patient_num.remove('00')
print(patient_num)
# img1data = np.asarray(img1.get_fdata())
# img2data = np.asarray(img2.get_fdata())
#
# diff = img2data - img1data
# # diff = binary_erosion(diff, structure=np.ones((2, 2, 2)))
#
# extract_seg_errors_rt_remove_empty.py
# extract_seg_errors_rt.pyprint(diff.shape)


# def show_slices(slices):
#     fig, axes = plt.subplots(1, len(slices))
#     for i, slice in enumerate(slices):
#         axes[i].imshow(slice.T, cmap="gray", origin="lower")
#
#
# slice_0 = diff[324, :, :]
# slice_1 = diff[:, 345, :]
# slice_2 = diff[:, :, 76]
# show_slices([slice_0, slice_1, slice_2])
# #
# plt.show()
#s = generate_binary_structure(2, 2)
# print(struct)
# s = iterate_structure(struct, 3).astype(int)
# print(s)
# labeled_array, num_features = label(diff, structure=s)

# print(labeled_array)
# x = find_objects(labeled_array)
# print(x)
# print(num_features)


# for num_slice in range(0, diff.shape[2]):
#     temp_diff = diff[:, :, num_slice]
#     labeled_array, num_features = label(temp_diff, structure=s)
#
#     for i in range(0, num_features+1):
#         if len(np.argwhere(labeled_array == i)) < 10:
#             temp_diff[labeled_array == i] = 0
#     diff[:, :, num_slice] = temp_diff
#

# diff_path = 'UCL_LUNG_01/Heart_diff_01.nii'
# diff_nii = nib.Nifti1Image(diff, img1.affine, img1.header)
# nib.save(diff_nii, diff_path)
s = generate_binary_structure(2, 2)

organs = ['CanalPRV', 'Heart', 'LLung', 'RLung', 'Oesophagus', 'SpinalCanal', 'Lungs', 'Spinal Canal']
for number in range(len(patient_num)):
    sourcedir = '/Volumes/My Passport for Mac/Project/NewNift/UCL_LUNG_' + patient_num[number]
    a = os.listdir(sourcedir)
    destdir = os.makedirs('/Volumes/My Passport for Mac/Project/NewNift/Differences/'+ str(patient_num[number]))



    for x in organs:
        if os.path.isfile(sourcedir + '/' + x + 'Cor_' + patient_num[number] + '.nii') and os.path.isfile(
                sourcedir + '/' + x + 'Er_' + patient_num[number] + '.nii'
        ):
            img1 = nib.load(sourcedir + '/' + x + 'Cor_' + patient_num[number] + '.nii')
            img2 = nib.load(sourcedir + '/' + x + 'Er_' + patient_num[number] + '.nii')
            img1data = np.asarray(img1.get_fdata())
            img2data = np.asarray(img2.get_fdata())
            diff = img2data - img1data
            for num_slice in range(0, diff.shape[2]):
                temp_diff = diff[:, :, num_slice]
                labeled_array, num_features = label(temp_diff, structure=s)

                for i in range(0, num_features + 1):
                    if len(np.argwhere(labeled_array == i)) < 10:
                        temp_diff[labeled_array == i] = 0
                diff[:, :, num_slice] = temp_diff
            diff_path = '/Volumes/My Passport for Mac/Project/NewNift/Differences/'+ str(patient_num[number]) + '/' + str(x) + '_Diff_' + str(
                patient_num[number]) + '.nii'
            diff_nii = nib.Nifti1Image(diff, img1.affine, img1.header)
            nib.save(diff_nii, diff_path)