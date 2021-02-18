import PIL
import numpy as np
import glob
from demonsReg import runRegistration
from funcsForCoursework import read_file, resampImageWithDefField, save_file, array_to_image

testing_dir = 'testing/'
template = read_file('Prior_Dir/Average_Template.png')
seg1 = read_file('Prior_Dir/GM.png')
seg2 = read_file('Prior_Dir/WM.png')
seg3 = read_file('Prior_Dir/CSF.png')
seg4 = read_file('Prior_Dir/NonBrain.png')

for test_img in glob.glob(testing_dir + '*image*'):
    test_img_name = test_img.replace('.png', '').replace('testing\\', '')
    test_img = read_file(test_img)

    image, field = runRegistration(template, test_img, disp_final_result=False, sigma_fluid= 5)

    prior1 = resampImageWithDefField(seg1, field)
    prior2 = resampImageWithDefField(seg2, field)
    prior3 = resampImageWithDefField(seg3, field)
    prior4 = resampImageWithDefField(seg4, field)
    print(test_img_name)

output = array_to_image(abs(seg1 - prior1))
output.show()

