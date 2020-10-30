
import sys
sys.path.append('/media/ducanh/DATA/tienln/ai_camera/ai_camera_detector')
from model.mb_ssd_lite_f19 import create_mb_ssd_lite_f19, create_mb_ssd_lite_f19_predictor
from model.mb_ssd_lite_f38 import create_mb_ssd_lite_f38, create_mb_ssd_lite_f38_predictor
from model.mb_ssd_lite_f38_person import create_mb_ssd_lite_f38_person, create_mb_ssd_lite_f38_person_predictor
from model.rfb_tiny_mb_ssd import create_rfb_tiny_mb_ssd, create_rfb_tiny_mb_ssd_predictor

from utils.misc import Timer
from torchscope import scope
import argparse
import cv2
import sys
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector predictor With Pytorch')
parser.add_argument("--net_type", default="rfb_tiny_mb2_ssd", type=str,help='mb2-ssd-lite_f19, mb2-ssd-lite_f38, rfb_tiny_mb2_ssd')
parser.add_argument('--model_path', default = '/media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/app/person/rfb_tiny_mb2_ssd_c32/rfb_tiny_mb2_ssd_c32_63_208_222.pth',
                     help='model weight')
parser.add_argument('--label_path', default = '/media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/utils/labels/person.txt', help='class names lable')
parser.add_argument('--result_path', default = 'detect_results', help='result path to save')
parser.add_argument('--test_path', default = "/media/ducanh/DATA/tienln/data/test_data/mall", help='path of folder test')
parser.add_argument('--test_device', default="cuda:0", type=str,help='cuda:0 or cpu')
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
    scope(net, (3, 300, 300))
    return predictor

if __name__ == "__main__":
    tt_time = 0
    predictor = load_model()

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    listdir = os.listdir(args.test_path)
    sum = 0

    for image_path in listdir:
        orig_image = cv2.imread(os.path.join(args.test_path, image_path))
        # orig_image = cv2.resize(orig_image, (640,480))
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        import time
        t1 = time.time()
        boxes, labels, probs = predictor.predict(image, 2000,0.5)
        tt_time += (time.time()-t1)
        probs = probs.numpy()
        sum += boxes.size(0)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
            cv2.putText(orig_image, str(probs[i]), (box[0], box[1]+20),cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255)) 
        cv2.putText(orig_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(args.result_path, image_path), orig_image)
        print(f"Found {len(probs)} object. The output image is {args.result_path}")
    print(sum, tt_time/36) #101002540945