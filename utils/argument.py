import argparse
from utils.misc import str2bool
import logging
import sys
def _argument():
    parser = argparse.ArgumentParser(
                        description='Single Shot MultiBox Detector Training With Pytorch')

    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and open_images.')

    parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
    parser.add_argument('--validation_dataset', help='Dataset directory path')
    parser.add_argument('--balance_data', action='store_true',
                        help="Balance training data by down-sampling more frequent labels.")

    parser.add_argument('--net', default="mb2-ssd-lite_f38_person",
                        help="It can be mb2-ssd-lite_f19, mb2-ssd-lite_f38,mb2-ssd-lite_f38_person, rfb_tiny_mb2_ssd")
    parser.add_argument('--freeze_base_net', action='store_true',
                        help="Freeze base net layers.")
    parser.add_argument('--freeze_net', action='store_true',
                        help="Freeze all the layers except the prediction head.")

    parser.add_argument('--mb2_width_mult', default=1, type=float,
                        help='Width Multiplifier for MobilenetV2')

    # Params for SGD
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--base_net_lr', default=None, type=float,
                        help='initial learning rate for base net.')
    parser.add_argument('--extra_layers_lr', default=None, type=float,
                        help='initial learning rate for the layers not in base net and prediction heads.')


    # Params for loading pretrained basenet or checkpoints.
    parser.add_argument('--base_net',
                        help='Pretrained base model')
    parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')

    # Scheduler
    parser.add_argument('--scheduler', default="multi-step", type=str,
                        help="Scheduler for SGD. It can one of multi-step and cosine")

    # Params for Multi-step Scheduler
    parser.add_argument('--milestones', default="80,100", type=str,
                        help="milestones for MultiStepLR")

    # Params for Cosine Annealingmodels/mb2-ssd-lite-epoch-8-train_loss-2.88-val_loss-2.6.pth
    parser.add_argument('--t_max', default=120, type=float,
                        help='T_max value for Cosine Annealing Scheduler.')

    # Train params
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', default=500, type=int,
                        help='the number epochs')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--validation_epochs', default=1, type=int,
                        help='the number epochs')
    parser.add_argument('--debug_steps', default=20, type=int,
                        help='Set the debug log output frequency.')
    parser.add_argument('--use_cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--valid', default=True, type=bool,
                        help='valid when training')                   

    parser.add_argument('--checkpoint_folder', default='/media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/weight/training',
                        help='Directory for saving checkpoint models')

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parser.parse_args()

    return args