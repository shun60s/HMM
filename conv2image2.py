import os
import numpy as np
import cv2

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy 1.14.0 
#  opencv-python 2.4.13.5

# Whole data write

def conv2image2(spec, file_name='out3.pmg', DEF_MAX=45.0,DEF_MIN=-45.0, Fshow=True):
    # create a random image
    num_banks=spec.shape[1]
    #print ('num_banks', num_banks)
    image=np.array(bytearray(os.urandom(  spec.shape[0] * num_banks))) 
    image=image.reshape(spec.shape[0],num_banks)
    
    # convert to gray data 0-255 between DEF_MAX and DEF_MIN
    spec2= (1 + ( spec - DEF_MAX )/( DEF_MAX - DEF_MIN)) *255
    
    np.putmask(spec2, spec2 > 255, 255)
    np.putmask(spec2, spec2 < 0, 0)
    spec3= spec2.astype(np.uint8)
    
    for i in range ( spec3.shape[0]  ):
        image[i]=spec3[i]
    
    #
    #
    image=np.rot90(image)
    
    if Fshow :
        cv2.imshow('Image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cv2.imwrite( file_name ,image)