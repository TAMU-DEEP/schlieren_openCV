from openpiv import tools, process, validation, filters, scaling 
import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import time

#cap = cv2.VideoCapture(0)
#
#time.sleep(2)
#ret, frame_a = cap.read()
##
#time.sleep(10)
#ret, frame_b = cap.read()
#frame_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
#frame_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)

frame_a  = tools.imread( '/Users/ryandmueller/Documents/schlieren_openCV/8D5A0666.tif' )
frame_b  = tools.imread( '/Users/ryandmueller/Documents/schlieren_openCV/8D5A0667.tif' )


print (frame_a)
print (frame_b)


fig,ax = plt.subplots(1,2)
ax[0].imshow(frame_a,cmap=plt.cm.gray)
ax[1].imshow(frame_b,cmap=plt.cm.gray)



winsize = 24 # pixels
searchsize = 64  # pixels, search in image B
overlap = 12 # pixels
dt = 0.02 # sec


u0, v0, sig2noise = process.extended_search_area_piv( frame_a.astype(np.int32), frame_b.astype(np.int32), window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method='peak2peak' )

x, y = process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap )

u1, v1, mask = validation.sig2noise_val( u0, v0, sig2noise, threshold = 1.3 )


u2, v2 = filters.replace_outliers( u1, v1, method='localmean', max_iter=10, kernel_size=2)

x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor = 10)

tools.save(x, y, u3, v3, mask, 'exp1_001.txt' )

tools.display_vector_field('exp1_001.txt', scale=100, width=0.005)
