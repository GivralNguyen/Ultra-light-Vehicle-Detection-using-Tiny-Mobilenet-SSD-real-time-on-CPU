# from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
# from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
# from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
# from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
# from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import os 
import imghdr

from evaluator.evalua import metrics, show_metrics
from crop.crop_img import random_crop, load_json
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
sys.path.append('/home/quannm/Documents/code/TinyMBSSD_Vehicle/')
from model.rfb_tiny_mb_ssd import create_rfb_tiny_mb_ssd, create_rfb_tiny_mb_ssd_predictor
import timeit
from utils.misc import Timer
from torchscope import scope
import argparse
import cv2
import sys
import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector predictor With Pytorch')
parser.add_argument("--net_type", default="rfb_tiny_mb2_ssd", type=str,help='rfb_tiny_mb2_ssd')
parser.add_argument('--model_path', default = '/home/quannm/Documents/code/TinyMBSSD_Vehicle/checkpoint/rfb_tiny_mb2_ssd-epoch-33-train_loss-0.95-val_loss-1.39.pth',
                     help='model weight')
parser.add_argument('--label_path', default = '/home/quannm/Documents/code/TinyMBSSD_Vehicle/utils/labels/vehicle.txt', help='class names lable')
parser.add_argument('--result_path', default = '/home/quannm/Documents/code/TinyMBSSD_Vehicle/detect_results', help='result path to save')
parser.add_argument('--test_path', default = "/home/quannm/Documents/code/TinyMBSSD_Vehicle/test_data", help='path of folder test')
parser.add_argument('--test_device', default="cpu", type=str,help='cuda:0 or cpu')
args = parser.parse_args()

def load_model():

    class_names = [name.strip() for name in open(args.label_path).readlines()]
    if args.net_type == 'mb2-ssd-lite_f19':
        net = create_mb_ssd_lite_f19(len(class_names), is_test=True)
        net.load(args.model_path)
        predictor = create_mb_ssd_lite_f19_predictor(net, candidate_size=200)
    elif args.net_type == 'mb2-ssd-lite_f38':
        net = create_mb_ssd_lite_f38(len(class_names), is_test=True, )
        predictor = create_mb_ssd_lite_f38_predictor(net, candidate_size=2000)
        net.load(args.model_path)
    elif args.net_type == 'mb2-ssd-lite_f38_person':
        net = create_mb_ssd_lite_f38_person(len(class_names), is_test=True, )
        predictor = create_mb_ssd_lite_f38_person_predictor(net, candidate_size=2000)
        net.load(args.model_path)
    elif args.net_type == 'rfb_tiny_mb2_ssd':
        net = create_rfb_tiny_mb_ssd(len(class_names), is_test=True, device=args.test_device)
        net.load(args.model_path)
        predictor = create_rfb_tiny_mb_ssd_predictor(net, candidate_size=5000, device=args.test_device)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)
    scope(net, (3, 320, 240))
    return predictor

if __name__ == "__main__":
    tt_time = 0
    
    sys.path.append("/home/minhhoang/Desktop/Detector/person-detection/ssdbenchmark")
    from evaluate.evalua import metrics, show_metrics
    from evaluate import random_crop, load_json

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    net_type = "mb2-ssd-lite"
    class_names = ['backdround', 'car', 'bus']
    num_classes = len(class_names)
    predictor = load_model()
    print('---------model loaded sucessful----------------')

    # day la test
    path_images = open("/home/quannm/Documents/code/TinyMBSSD_Vehicle/img.txt").read().splitlines()
    path_json = open("/home/quannm/Documents/code/TinyMBSSD_Vehicle/json.txt").read().splitlines()
    dir_img = "/media/minhhoang/Data/dataPerson/"
    num_img = len(path_json)
    count = 0
    lst_gts = []
    lst_dts = []
    c = ''
    import time
    lst_time = []
    p_save_img = "/home/quannm/Documents/code/TinyMBSSD_Vehicle/detect_results"

    if not os.path.exists(p_save_img):
        os.makedirs(p_save_img)
    sum_dt = 0
    for image_path, json_path in zip(path_images, path_json):
        print(count)
        image_path = image_path

        
        if count == 132:
            print("abcd")
        name_img = image_path.split("/")[-1]
        type_img = imghdr.what(image_path)
        if type_img == "jpeg" or type_img == None:
            type_img = "jpg"
        name_img, _ = name_img.split(".")
        orig_image = cv2.imread(image_path)
        json_path = json_path
        image, gts, img_gt = random_crop(image_path, json_path, p_save_img, crop=False, new_size = (640,480),resize_crop=False, resize = (640,480))

        start = time.time()
        boxes, labels, probs = predictor.predict(image, -1, 0.5)
        lst_time.append(time.time()-start)
        dts = []
        sum_dt += boxes.size(0)
        for i in range(boxes.size(0)):

            dt = []
            dt.append(name_img)
            label = 1
            box = boxes[i, :]
            dt.append(class_names[labels[i]])
            dt.append(float(probs[i]))
            box = boxes[i, :]
            score = probs[i]
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])
            tup = tuple([x_min, y_min, x_max, y_max])
            dt.append(tup)
            dts.append(dt)
            label = f"{class_names[labels[i]]}: {probs[i]:.5f}"
            cv2.rectangle(img_gt, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

            cv2.putText(img_gt, label,
                        (box[0]+20, box[1]+20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # font scale
                        (0, 0, 255),
                        1)  # line type
        cv2.imwrite(p_save_img+name_img+"_"+str(count)+".jpg", img_gt)
        lst_gts += gts
        lst_dts += dts
        count += 1
        if count == 10:
            break

    save_mr = "/home/quannm/Documents/code/TinyMBSSD_Vehicle/detect_results/space_img.json"
    save_miss_wh = "/home/quannm/Documents/code/TinyMBSSD_Vehicle/detect_results/miss_wh.txt"

    print("Num image: ",count)
    metric = metrics(lst_gts, lst_dts, class_names, save_miss=save_miss_wh, save_mr=save_mr)
    show_metrics(metric, count)
    print("Aver Proces:", sum(lst_time)/ len(lst_time))