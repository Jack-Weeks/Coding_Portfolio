import numpy as np
import random
import PIL

# emptylist = [x1,x2,x3]
# print(emptylist.index(max(emptylist)))
img_dims_y = 200
img_dims_x = 200
output1 = np.zeros((img_dims_y, img_dims_x), dtype='float64')
output2 = np.zeros((img_dims_y, img_dims_x), dtype='float64')
output3 = np.zeros((img_dims_y, img_dims_x), dtype='float64')
for j in range(200):
    for i in range(200):
        x1 = random.random()
        x2 = random.random()
        x3 = random.random()
        LNCCs = [x1, x2, x3]
        print(LNCCs.index(max(LNCCs)))
        if LNCCs.index(max(LNCCs)) == 0 and LNCCs[0] > 0.33:
            output1[j,i] = 255
            output2[j,i] = 0
            output3[j,i] = 0
        elif LNCCs.index(max(LNCCs)) == 1 and LNCCs[1] > 0.33:
            output1[j,i] = 0
            output2[j,i] = 255
            output3[j,i] = 0
        elif LNCCs.index(max(LNCCs)) == 2 and LNCCs[2] > 0.33:
            output1[j,i] = 0
            output2[j,i] = 0
            output3[j,i] = 255
        else:
            output1[j,i] = 0
            output2[j,i] = 0
            output3[j,i] = 0



