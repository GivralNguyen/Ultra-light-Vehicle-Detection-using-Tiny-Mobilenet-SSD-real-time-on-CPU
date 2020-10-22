import sys
sys.path.append('/media/ducanh/DATA/tienln/ai_camera/detector/')
from model.mb_ssd_lite_f19 import create_mb_ssd_lite_f19, create_mb_ssd_lite_f19_predictor
from model.mb_ssd_lite_f38 import create_mb_ssd_lite_f38, create_mb_ssd_lite_f38_predictor
from model.rfb_tiny_mb_ssd import create_rfb_tiny_mb_ssd, create_rfb_tiny_mb_ssd_predictor
import argparse
import torch

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector predictor With Pytorch')
parser.add_argument("--net_type", default="rfb_tiny_mb2_ssd", type=str,help='mb2-ssd-lite_f19, mb2-ssd-lite_f38, rfb_tiny_mb2_ssd')
parser.add_argument('--model_path', default = '/media/ducanh/DATA/tienln/ai_camera/tiny_ssd/models/train_model/Epoch-146-loss-1.42-val-1.9.pth',help='model weight')
parser.add_argument('--label_path', default = '/media/ducanh/DATA/tienln/ai_camera/detector/utils/labels/person.txt', help='class names lable')
args = parser.parse_args()

num_classes = len([name.strip() for name in open(args.label_path).readlines()])

if args.net_type == 'mb2-ssd-lite_f19':
    net = create_mb_ssd_lite_f19(num_classes)
elif args.net_type == 'mb2-ssd-lite_f38':
    net = create_mb_ssd_lite_f38_predictor(num_classes)
elif args.net_type == 'rfb_tiny_mb2_ssd':
    net = create_rfb_tiny_mb_ssd(num_classes)
else:
    print("unsupport network type.")
    sys.exit(1)
net.load(args.model_path)
net.eval()
net.to("cuda")

model_name = args.model_path.split("/")[-1].split(".")[0]
model_path = f"app/person/{model_name}.onnx"

dummy_input = torch.randn(1, 3, 240, 320).to("cuda")
torch.onnx.export(net, dummy_input, model_path, verbose=False, input_names=['input'], output_names=['scores', 'boxes'])