import numpy as np
import torch

image_size = [320, 240]
image_mean_test = image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2

def generate_priors(size):
    shrinkage_list = []
    priors = []
    min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    feature_map_list = [[40, 20, 10, 5], [30, 15, 8, 4]]
    ratios = [1.7, 1.9, 2.1]
    for i in range(0, len(image_size)):
        item_list = []
        for k in range(0, len(feature_map_list[i])):
            item_list.append(image_size[i] / feature_map_list[i][k])
        shrinkage_list.append(item_list)
    
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    for ratio in ratios:
                        w = min_box / image_size[0]
                        h = min_box / image_size[1]
                        priors.append([
                            x_center,
                            y_center,
                            w,
                            h*ratio
                        ])
    # print("priors nums:{}".format(len(priors)))
    priors = torch.tensor(priors)
    torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors
priors = generate_priors(320)
