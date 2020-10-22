import sys
sys.path.append('/media/ducanh/DATA/tienln/ai_camera/detector/')
import cv2
# from PIL import Image
import numpy as np
import os
from datasets.data_preprocessing import PredictionTransform

size = 300
transform = PredictionTransform(size)

def get_input_FD(img_raw):
    img = transform(img_raw)
    img = np.expand_dims(img, axis=0)
    return img

image_path = '/media/ducanh/DATA/tienln/ai_camera/app/app_head_detection/imgs'
images = os.listdir(image_path)
paths = []
for image_name in images:
    print(image_name)
    img = cv2.imread(os.path.join(image_path, image_name))
    img_processed = get_input_FD(img)
    img_processed.tofile(os.path.join('/media/ducanh/DATA/tienln/ai_camera/detector/app/quantize/raw_data', image_name).replace('.jpg', ".raw"))
    paths.append(os.path.join('/media/ducanh/DATA/tienln/ai_camera/detector/app/quantize/raw_data', image_name).replace('.jpg', ".raw"))
    print("test")

with open('/media/ducanh/DATA/tienln/ai_camera/detector/app/quantize/data_quantize_person.txt', 'w') as f:
    for item in paths:
        f.write("%s\n" % item)