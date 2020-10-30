
from utils.argument import _argument
import logging
import sys
import itertools
from utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from datasets.data_loader import _DataLoader
from module.ssd import MatchPrior
from datasets.data_preprocessing import TrainAugmentation, TestTransform
from torch.utils.data import DataLoader, ConcatDataset
from utils.loss import MultiboxLoss, FocalLoss
from torchsummary import summary
import torch
from torchscope import scope
import sys
sys.path.append('/media/ducanh/DATA/tienln/ai_camera/detector/')
from utils.misc import str2bool, Timer, freeze_net_layers, store_labels
timer = Timer()

args = _argument()

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    training_loss = 0.0
    for i, data in enumerate(loader):
        print(".", end="", flush=True)
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % args.debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"train_avg_loss: {avg_loss:.4f}, " +
                f"train_reg_loss: {avg_reg_loss:.4f}, " +
                f"train_cls_loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            training_loss = avg_loss

    return training_loss

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1
        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

def data_loader(config):
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,config.size_variance, config.iou_threshold)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    Data_Train = []
    Data_Valid = []
    datasets = []

    path_dataset = open("/media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/datasets/train_dataset.txt", "r")
    for line in path_dataset:
        data = line.split('+')
        Data_Train.append([data[0],data[1][:-1]])
    
    # training datasets
    # dataset_paths = [Data_Train[0],Data_Train[1],Data_Train[2],Data_Train[3],Data_Train[4],Data_Train[5]]
    dataset_paths = [Data_Train[3]]
    for dataset_path in dataset_paths:
        print(dataset_path)
        dataset = _DataLoader(dataset_path, transform=train_transform,target_transform=target_transform)
        print(len(dataset.ids))
        datasets.append(dataset)
        num_classes = len(dataset.class_names)
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,num_workers=args.num_workers,shuffle=True)

    if args.valid:
        # Validation datasets
        path_dataset = open("/media/ducanh/DATA/tienln/ai_camera/ai_camera_detector/datasets/valid_dataset.txt", "r")
        for line in path_dataset:
            data = line.split('+')
            Data_Valid.append([data[0],data[1][:-1]])
        # print(Data_Valid)
        logging.info("Prepare Validation datasets.")
        valid_dataset_paths = [Data_Valid[0]]
        for dataset_path in valid_dataset_paths:
            val_dataset = _DataLoader(dataset_path, transform=test_transform,target_transform=target_transform)
        val_loader = DataLoader(val_dataset, args.batch_size,num_workers=args.num_workers,shuffle=True)
        return train_loader, val_loader, num_classes
    else:
        return train_loader, num_classes

def create_network(create_net,num_classes, DEVICE ):
    logging.info("Build network.")
    net = create_net(num_classes)
    # print(net)
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    # criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
    #                          center_variance=0.1, size_variance=0.2, device=DEVICE)
    criterion = FocalLoss()
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    return net, criterion, optimizer, scheduler
