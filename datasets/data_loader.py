import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os
import json
import sys
class _DataLoader:

    def __init__(self, root, transform=None, target_transform=None):

        self.anno_path = root[0]
        self.img_path = root[1]

        self.transform = transform
        self.target_transform = target_transform
        self.ids = []
        self.class_names = ('BACKGROUND','car','bus','motorcycle','bicycle','truck')
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        
        self._annopath = os.path.join('%s', 'json_annotations', '%s.json')

        for file in os.listdir(self.anno_path):
            with open(os.path.join(self.anno_path,file), 'r') as f:
                data = json.load(f)
            objects = data["objects"]
            for sub_object in data["objects"]:
                if sub_object["label"]== "car" or "bus" or "van" or "motorcycle" or "bicycle" or "truck" :
                    #print(file.split(".json")[0])
                    self.ids.append(file.split(".json")[0])
                    break
        print("finished loading")
                 

    def __getitem__(self, index):
        image_id = self.ids[index]
        #print(image_id)
        boxes, labels= self._get_annotation(image_id)
        #print(boxes)
        image = self._read_image(image_id)
        #print(image)
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
            #print(image)
            #print(image)
        if self.target_transform:
            
            boxes_orig,labels_orig = boxes,labels 
            #print(boxes_orig)
            #print(labels_orig)
            boxes, labels = self.target_transform(boxes, labels)
            print(labels)
        return image, boxes, labels

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.anno_path,image_id+".json")
        # print(annotation_file)
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            #print(f)
        objects = data["objects"]
        boxes = []
        labels = []
        for sub_object in objects:
            flag = 0
            class_name = sub_object["label"]   
            if class_name == "van":
                #print('converted van to car')
                flag = 1
                class_name = 'car'    
            #print(class_name)    
            if class_name in self.class_dict:
                #print("Flag is: ",flag)
                bbox = sub_object["bbox"]
                #print(type(bbox))
                if type(bbox) == list:
                    
                    x1 = float(bbox[0])
                    y1 = float(bbox[1])
                    x2 = x1 + float(bbox[2])
                    y2 = y1 + float(bbox[3])
                else:
                    #print('hey')
                    x1 = float(bbox["x_topleft"])
                    y1 = float(bbox["y_topleft"])
                    x2 = x1 + float(bbox["w"])
                    y2 = y1 + float(bbox["h"])
         
                #print(x1,y1,x2,y2)
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_dict[class_name])
                #if (self.class_dict[class_name] == 4):
                #    print('label appended: ',self.class_dict[class_name] , class_name)
                #print(np.array(labels, dtype=np.int64))
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def _read_image(self, image_id):
        if os.path.isfile(os.path.join(self.img_path,image_id+".jpg")):
            image_file = os.path.join(self.img_path,image_id+".jpg")
        elif os.path.isfile(os.path.join(self.img_path,image_id+".jpeg")):
            image_file = os.path.join(self.img_path,image_id+".jpeg")
        else :
            image_file = os.path.join(self.img_path,image_id+".png")
        image = cv2.imread(str(image_file))
        if image is None:
            print("the image is not available", image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image