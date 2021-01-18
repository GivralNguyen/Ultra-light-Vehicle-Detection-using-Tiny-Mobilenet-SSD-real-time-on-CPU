import cv2
from keras.preprocessing import image
import json
import os
import random
import glob
def load():
    gts = []
    with open(path_file) as f:
        fh1 = json.load(f)
        _ , nameOfImage = nameOfImage.split("-")
        for line in fh1['objects']:
            gt = []
            gt.append(nameOfImage)
            # print(line['label'])
            gt.append(line['label'])
            gt.append(1)
            x = int(line['bbox']['x_topleft']) #x_topleft
            y = int(line['bbox']['y_topleft']) #y_topleft
            w = int(line['bbox']['w'])
            h = int(line['bbox']['h'])
            if x > box_img[1] and y > box_img[0] and (x+w)<(box_img[1]+box_img[3]) and (y+h)<(box_img[2]+box_img[0]):
                xnew = x - box_img[1]
                ynew = y - box_img[0]
                wnew = w
                hnew = h
                tup = tuple([xnew, ynew, xnew+w, ynew+h])
                print(tup)
                gt.append(tup)
                # print(xnew, ynew, wnew, hnew)
                cv2.rectangle(img, (xnew, ynew), (xnew + wnew, ynew + hnew), (0, 255, 0), 2)
                gts.append(gt)