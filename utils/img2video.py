	
import cv2
import numpy as np
import glob
 

img_array = []  
foldername= '/home/quannm/Documents/code/ai_camera_detectorcopy/detect_results/'
for i in range (9000,13413):
    filename = foldername+str(i)+str('.jpg')
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
out = cv2.VideoWriter('dongkinhnghiathuc.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()