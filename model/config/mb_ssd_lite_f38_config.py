
from typing import List
import numpy as np
import torch
import itertools
import math
import collections

image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.6
center_variance = 0.1
size_variance = 0.2

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])
SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])

specs = [
    SSDSpec(38, 8, SSDBoxSizes(16, 32), [1.7]),
    SSDSpec(19, 16, SSDBoxSizes(32, 64), [1.7]),
    SSDSpec(10, 32, SSDBoxSizes(64, 128), [1.7]),
    SSDSpec(5, 64, SSDBoxSizes(128, 256), [1.7]),
]
def generate_ssd_priors(specs: List[SSDSpec], image_size, clamp=True) -> torch.Tensor:
    priors = []
    for spec in specs:
        scale = image_size / spec.shrinkage
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale
            y_center = (j + 0.5) / scale
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio)

                # small sized square box
                size = spec.box_sizes.min
                h = w = size / image_size
                priors.append([
                    x_center,
                    y_center,
                    w,
                    h*ratio
                ])
                # big sized square box
                size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
                h = w = size / image_size
                priors.append([
                    x_center,
                    y_center,
                    w,
                    h*ratio
                ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors

priors = generate_ssd_priors(specs, image_size)