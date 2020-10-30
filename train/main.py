import sys
sys.path.append('/media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/')
from utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import os
import logging
import sys
import itertools
import torch
from torchscope import scope
from torchsummary import summary
from utils.loss import MultiboxLoss, FocalLoss
from utils.argument import _argument
from train import train, test, data_loader, create_network
from model.mb_ssd_lite_f38 import create_mb_ssd_lite_f38
from model.config import mb_ssd_lite_f38_config

from model.mb_ssd_lite_f38_person import create_mb_ssd_lite_f38_person
from model.config import mb_ssd_lite_f38_person_config

from model.mb_ssd_lite_f19 import create_mb_ssd_lite_f19
from model.config import mb_ssd_lite_f19_config
from model.rfb_tiny_mb_ssd import create_rfb_tiny_mb_ssd
from model.config import rfb_tiny_mb_ssd_config

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Train():
    '''
    The class to training
    '''
    def __init__(self):
        self.args = _argument()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
        self.net, self.criterion, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.get_model()
        self.dir_path = os.path.join(self.args.checkpoint_folder,self.args.net)
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

    def get_model(self):
        timer = Timer()
        logging.info(self.args)
        
        if self.args.net == 'mb2-ssd-lite_f19':
            create_net = create_mb_ssd_lite_f19
            config =  mb_ssd_lite_f19_config
        elif self.args.net == 'mb2-ssd-lite_f38':
            create_net = create_mb_ssd_lite_f38
            config = mb_ssd_lite_f38_config
        elif self.args.net == 'mb2-ssd-lite_f38_person':
            create_net = create_mb_ssd_lite_f38_person
            config = mb_ssd_lite_f38_person_config
        elif self.args.net == 'rfb_tiny_mb2_ssd':
            create_net = create_rfb_tiny_mb_ssd
            config = rfb_tiny_mb_ssd_config
        else:
            logging.fatal("The net type is wrong.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        train_loader,val_loader, num_classes = data_loader(config)
        net, criterion, optimizer, scheduler = create_network(create_net, num_classes, self.device)

        return net, criterion, optimizer, scheduler, train_loader, val_loader

    def training (self):
        print(self.dir_path)
        for epoch in range(0, self.args.num_epochs):
            self.scheduler.step()
            training_loss = train(self.train_loader, self.net, self.criterion, self.optimizer, device=self.device, debug_steps=self.args.debug_steps, epoch=epoch)
            if epoch % self.args.validation_epochs == 0 or epoch == self.args.num_epochs - 1:
                if self.args.valid:
                    val_running_loss, val_running_regression_loss, val_running_classification_loss = test(self.val_loader,self.net,self.criterion,device=self.device)
                    logging.info(
                        f"Epoch: {epoch}, " +
                        f"val_avg_loss: {val_running_loss:.4f}, " +
                        f"val_reg_loss {val_running_regression_loss:.4f}, " +
                        f"val_cls_loss: {val_running_classification_loss:.4f}")
                    model_path = os.path.join(self.dir_path, f"{self.args.net}-epoch-{epoch}-train_loss-{round(training_loss,2)}-val_loss-{round(val_running_loss,2)}.pth")
                else :
                    model_path = os.path.join(self.dir_path, f"{self.args.net}-epoch-{epoch}-train_loss-{round(training_loss,2)}.pth")
                self.net.save(model_path)
                logging.info(f"Saved model {self.dir_path}")

if __name__ == '__main__':
    train = Train().training()