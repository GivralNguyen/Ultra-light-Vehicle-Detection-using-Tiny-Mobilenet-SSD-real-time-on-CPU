from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d

from module.rfb_tiny_mobilenet_v2 import Mb_Tiny_RFB
from model.config import rfb_tiny_mb_ssd_config as config
from utils.predictor import Predictor
from module.ssd import SSD


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_rfb_tiny_mb_ssd(num_classes, is_test=False, device="cuda"):
    base_net = Mb_Tiny_RFB(2)
    base_net_model = base_net.model  # disable dropout layer

    source_layer_indexes = [8,11,13]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=base_net.base_channel * 16, out_channels=base_net.base_channel * 4, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=base_net.base_channel * 16, kernel_size=3, stride=2, padding=1),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=9 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=base_net.base_channel * 16, out_channels=9 * 4, kernel_size=3, padding=1)
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=9 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 8, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=base_net.base_channel * 16, out_channels=9 * num_classes, kernel_size=3, padding=1)
    ])

    return SSD(num_classes, base_net_model, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config, device=device)


def create_rfb_tiny_mb_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean_test,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
