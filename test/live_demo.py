import sys
sys.path.append('/media/ducanh/DATA/tienln/ai_camera/ai_camera_detector')
from model.mb_ssd_lite_f19 import create_mb_ssd_lite_f19, create_mb_ssd_lite_f19_predictor
from model.mb_ssd_lite_f38 import create_mb_ssd_lite_f38, create_mb_ssd_lite_f38_predictor
from model.rfb_tiny_mb_ssd import create_rfb_tiny_mb_ssd, create_rfb_tiny_mb_ssd_predictor
from utils.misc import Timer
import time
import argparse
import cv2
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector predictor With Pytorch')
parser.add_argument("--net_type", default="rfb_tiny_mb2_ssd", type=str,help='mb2-ssd-lite_f19, mb2-ssd-lite_f38, rfb_tiny_mb2_ssd')
parser.add_argument('--model_path', default = 'app/person/rfb_tiny_mb2_ssd_c32/rfb_tiny_mb2_ssd_c32_63_208_222.pth',
                     help='model weight')
parser.add_argument('--label_path', default = '/media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/utils/labels/person.txt', help='class names lable')
parser.add_argument('--result_path', default = 'results', help='result path to save')
parser.add_argument('--test_path', default = "mall", help='path of folder test')
parser.add_argument('--test_device', default="cpu", type=str,help='cuda:0 or cpu')
args = parser.parse_args()
# mb2-ssd-lite_FPN38-epoch-150-train_loss-1.93-val_loss-1.4
# 'rtsp://root:123456@192.168.0.241/axis-media/media.3gp'
capture = cv2.VideoCapture(0)
class_names = [name.strip() for name in open(args.label_path).readlines()]
num_classes = len(class_names)

def load_model():

    class_names = [name.strip() for name in open(args.label_path).readlines()]
    if args.net_type == 'mb2-ssd-lite_f19':
        net = create_mb_ssd_lite_f19(len(class_names), is_test=True)
        net.load(args.model_path)
        predictor = create_mb_ssd_lite_f19_predictor(net, candidate_size=200)
    elif args.net_type == 'mb2-ssd-lite_f38':
        net = create_mb_ssd_lite_f38(len(class_names), is_test=True, )
        predictor = create_mb_ssd_lite_f38_predictor(net, candidate_size=200)
        net.load(args.model_path)
    elif args.net_type == 'rfb_tiny_mb2_ssd':
        net = create_rfb_tiny_mb_ssd(len(class_names), is_test=True, device=args.test_device)
        net.load(args.model_path)
        predictor = create_rfb_tiny_mb_ssd_predictor(net, candidate_size=5000, device=args.test_device)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)

    return predictor

def live_demo():

    predictor = load_model()
    timer = Timer()
    count = 0
    while True:
        ret, orig_image = capture.read()
        count += 1
        if count ==15:
            name = int(time.time())
            cv2.imwrite('/media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/image_test_size/'+ str(name)+'.jpg', orig_image)
            count = 0
            print(name)
        orig_image = cv2.resize(orig_image, (480,360))
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
        timer.start()
        boxes, labels, probs = predictor.predict(image, 2000, 0.5)
        interval = timer.end()
        probs = probs.numpy()
        # print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            # cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # cv2.putText(orig_image, str(probs[i]), (box[0], box[1]+20),cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255)) 
        # cv2.putText(orig_image,"number of people: " + str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('annotated', orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_demo()