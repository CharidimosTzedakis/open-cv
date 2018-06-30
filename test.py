import sys
print '\n'.join(sys.path)
print '\n'.join(sys.modules)
import backports
print('Backports Path: {0}'.format(backports.__path__))

import cv2
import numpy as np
import matplotlib.pyplot as plt

# OpenCV ==> (B,G,R)
# plt ==> (R,G,B)

#Read Image
#img = cv2.imread('baboon.png')
img = cv2.imread('baboon.png', cv2.IMREAD_GRAYSCALE)
# IMREAD_COLOR = 1 
# IMREAD_UNCHANGED = -1 

#Display Image
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.plot([50,100],[80,100], 'c', linewidth=5)
plt.show()


#Applying Grayscale filter to image
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('image',gray)



#Saving filtered image to new file
#cv2.imwrite('graytest.jpg',gray)