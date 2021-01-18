
import sys
sys.path.append('/home/quannm/Documents/code/TinyMBSSD_Vehicle/')
from model.mb_ssd_lite_f19 import create_mb_ssd_lite_f19, create_mb_ssd_lite_f19_predictor
from model.mb_ssd_lite_f38 import create_mb_ssd_lite_f38, create_mb_ssd_lite_f38_predictor
from model.mb_ssd_lite_f38_person import create_mb_ssd_lite_f38_person, create_mb_ssd_lite_f38_person_predictor
from model.rfb_tiny_mb_ssd import create_rfb_tiny_mb_ssd, create_rfb_tiny_mb_ssd_predictor
import timeit
from utils.misc import Timer
#from torchscope import scope
import argparse
import cv2
import sys
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector predictor With Pytorch')
parser.add_argument("--net_type", default="rfb_tiny_mb2_ssd", type=str,help='mb2-ssd-lite_f19, mb2-ssd-lite_f38, rfb_tiny_mb2_ssd')
parser.add_argument('--model_path', default = '/home/quannm/Documents/code/ai_camera_detectorcopy/checkpoint_expand/rfb_tiny_mb2_ssd/rfb_tiny_mb2_ssd-epoch-33-train_loss-1.2-val_loss-1.31.pth',
                     help='model weight')
parser.add_argument('--label_path', default = '/home/quannm/Documents/code/ai_camera_detectorcopy/utils/labels/person.txt', help='class names lable')
parser.add_argument('--result_path', default = '/home/quannm/Documents/code/ai_camera_detectorcopy/detect_results', help='result path to save')
parser.add_argument('--test_path', default = "/media/quannm/DATAQUAN/Detrac/test/test_image/MVI_40701", help='path of folder test')
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
    #scope(net, (3, 300, 300))
    return predictor

if __name__ == "__main__":
    tt_time = 0
    predictor = load_model()

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    listdir = os.listdir(args.test_path)
    sum = 0
    
    k = 0
    j= 0
    for k in range (0,30000):
        print(os.path.join('/home/quannm/Documents/code/ai_camera_detectorcopy/test_data1/frame'+"{:01d}".format(k))+".jpg")
        orig_image = cv2.imread((os.path.join('/home/quannm/Documents/code/ai_camera_detectorcopy/test_data1/frame'+"{:01d}".format(k))+".jpg"))
        new_image = orig_image
        #print(orig_image.shape)
        # orig_image = cv2.resize(orig_image, (640,480))
        
        old_size = orig_image.shape[:2] 
        old_ratio = old_size[0]/old_size[1]
        #print(old_ratio)#0.5625
        new_height = old_size[0]
        new_width = old_size[1]
        #print(new_height)#540
        if old_ratio > 0.75:
            new_width = int(new_height * 4/3)
            #print(new_height,new_width)
            
        elif old_ratio < 0.75:
            new_height = int(new_width*3/4)
        
        #im = cv2.resize(im, (new_width, new_height))
            
        delta_h = new_height - old_size[0]
        delta_w = new_width - old_size[1]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_image = cv2.copyMakeBorder(orig_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color) 
            
        #new_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        import time
        
        
        boxes, labels, probs = predictor.predict(new_image, 2000,0.6)
        
        probs = probs.numpy()
        sum += boxes.size(0)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            box = box.numpy()
            
            if labels[i] == 1:
                class_name = 'car'
                cv2.rectangle(new_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 2)
                cv2.putText(new_image, str(probs[i]), (int(box[0]), int(box[1]+20)),cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255)) 
                cv2.putText(new_image, str(class_name), (int(box[0]-30), int(box[1]-10)),cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255))
            elif labels[i] == 2:
                class_name = 'bus'
            elif labels[i] == 3:
                class_name = 'motorcycle'  
            elif labels[i] == 4:
                class_name = 'bicycle'     
            elif labels[i] == 5:
                class_name = 'truck' 
           
        
        cv2.putText(new_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(os.path.join(args.result_path, image_path), new_image)
        cv2.imwrite(str(j)+str('.jpg'), new_image)
        j = j+1
        print(f"Found {len(probs)} object. The output image is {args.result_path}")
  